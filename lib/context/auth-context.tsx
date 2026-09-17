"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { apiFetch, clearStoredToken, getStoredToken, setStoredToken } from "@/lib/api/client";
import type { IdentityStatus } from "@/lib/types";
import {
  exchangeCodeForTokens,
  fetchAuthConfig,
  keycloakLogout,
  refreshKeycloakToken,
  startKeycloakLogin,
  type KeycloakPublicConfig,
} from "@/lib/api/keycloak";

/**
 * 身份信息（本地账号与 Keycloak 联邦账号共用同一结构）。
 *
 * tenant_id 就是"公司"（架构图里的 company_a / company_b），
 * role_label 是角色的中文名（企业管理员 / 知识库管理员 / 部门负责人 / 普通员工）。
 */
export interface AuthUser {
  id: string;
  username: string;
  role: string;
  display_name?: string | null;
  role_label?: string;
  tenant_id?: string;
  department_id?: string | null;
  auth_source?: "local" | "keycloak" | string;
  permissions?: string[];
  // ── 企业身份（个人主页展示 + 未验证拦截）─────────────────────────────────
  /** 公司名称原文（中文也合法，界面上显示这个而不是 tenant_id）。 */
  company_name?: string;
  department_name?: string | null;
  /** 部门职责 / 职务。 */
  job_title?: string | null;
  /** none | pending | approved | rejected */
  identity_status?: IdentityStatus;
}

interface AuthContextValue {
  user: AuthUser | null;
  /** False until the stored token has been checked once on mount. */
  hydrated: boolean;
  /** Keycloak 公开配置（null = 还没取到）。 */
  sso: KeycloakPublicConfig | null;
  login: (username: string, password: string) => Promise<AuthUser>;
  register: (username: string, password: string) => Promise<AuthUser>;
  /**
   * 企业统一身份登录：企业名称（= 邮箱）+ **已登记的企业职责** + 密码。
   * 由本系统账号体系校验（不跳转 Keycloak），职责与账号登记值不符则拒绝。
   */
  unifiedLogin: (
    username: string,
    duty: string,
    password: string
  ) => Promise<AuthUser>;
  /** 重新拉取身份信息（提交身份验证申请 / 审核通过后刷新状态用）。 */
  refreshUser: () => Promise<AuthUser | null>;
  /** 跳转企业统一身份登录（Keycloak + PKCE）。 */
  loginWithKeycloak: (returnTo?: string) => Promise<void>;
  /** OIDC 回调页调用：用 code 换令牌并建立会话。 */
  completeKeycloakLogin: (code: string, state: string) => Promise<string>;
  logout: () => void;
  /** 判断当前用户是否具备某项能力（与后端权限名一致）。 */
  can: (permission: string) => boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface TokenResponseRaw {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

const USER_KEY = "rag_user";
const REFRESH_KEY = "rag_refresh_token";
const ID_TOKEN_KEY = "rag_id_token";

function storeSession(token: string, user: AuthUser) {
  setStoredToken(token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [sso, setSso] = useState<KeycloakPublicConfig | null>(null);
  const refreshTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const applyMe = useCallback((me: AuthUser) => {
    setUser(me);
    localStorage.setItem(USER_KEY, JSON.stringify(me));
    return me;
  }, []);

  /**
   * 重新拉取 /auth/me 并刷新缓存.
   *
   * 身份验证申请提交 / 审核结论落地后必须调用它 —— 否则头像、导航栏、
   * 首页拦截读到的还是旧状态（提交完仍显示「去验证」，明明已经是「审核中」）。
   */
  const refreshUser = useCallback(async () => {
    try {
      const me = await apiFetch<AuthUser>("/auth/me", { timeout: 10000 });
      return applyMe(me);
    } catch {
      return null;
    }
  }, [applyMe]);

  // 1) 公开认证配置：决定登录页要不要显示"企业统一身份登录"
  useEffect(() => {
    fetchAuthConfig()
      .then(setSso)
      .catch(() => setSso(null));
  }, []);

  // 2) 恢复会话：本地缓存的用户先顶上（首屏不闪），再用 /auth/me 复核
  useEffect(() => {
    const token = getStoredToken();
    const cachedUser = localStorage.getItem(USER_KEY);

    if (!token) {
      setHydrated(true);
      return;
    }

    if (cachedUser) {
      try {
        setUser(JSON.parse(cachedUser) as AuthUser);
      } catch {
        localStorage.removeItem(USER_KEY);
      }
    }

    apiFetch<AuthUser>("/auth/me", { timeout: 10000 })
      .then((me) => {
        setUser(me);
        localStorage.setItem(USER_KEY, JSON.stringify(me));
      })
      .catch(() => {
        setUser(null);
        localStorage.removeItem(USER_KEY);
        clearStoredToken();
      })
      .finally(() => setHydrated(true));
  }, []);

  // 3) Keycloak 令牌静默续期：access token 默认 30 分钟，续期失败即视为登出
  useEffect(() => {
    if (!sso?.enabled) return;
    if (refreshTimer.current) clearInterval(refreshTimer.current);

    refreshTimer.current = setInterval(
      () => {
        const refreshToken = localStorage.getItem(REFRESH_KEY);
        if (!refreshToken || !sso) return;
        refreshKeycloakToken(sso, refreshToken)
          .then((tokens) => {
            setStoredToken(tokens.accessToken);
            if (tokens.refreshToken) {
              localStorage.setItem(REFRESH_KEY, tokens.refreshToken);
            }
            if (tokens.idToken) localStorage.setItem(ID_TOKEN_KEY, tokens.idToken);
          })
          .catch(() => {
            /* 续期失败不主动踢人：下一次请求 401 会统一跳登录页 */
          });
      },
      5 * 60 * 1000
    );

    return () => {
      if (refreshTimer.current) clearInterval(refreshTimer.current);
    };
  }, [sso]);

  const login = useCallback(
    async (username: string, password: string) => {
      const res = await apiFetch<TokenResponseRaw>("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      storeSession(res.access_token, res.user);
      setUser(res.user);
      return res.user;
    },
    []
  );

  const register = useCallback(
    async (username: string, password: string) => {
      const res = await apiFetch<TokenResponseRaw>("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      storeSession(res.access_token, res.user);
      setUser(res.user);
      return res.user;
    },
    []
  );

  const unifiedLogin = useCallback(
    async (username: string, duty: string, password: string) => {
      const res = await apiFetch<TokenResponseRaw>("/auth/unified-login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, duty, password }),
      });
      storeSession(res.access_token, res.user);
      setUser(res.user);
      return res.user;
    },
    []
  );

  const loginWithKeycloak = useCallback(
    async (returnTo = "/dashboard") => {
      const config = sso ?? (await fetchAuthConfig());
      setSso(config);
      if (!config.enabled) {
        throw new Error("企业统一身份登录未启用");
      }
      await startKeycloakLogin(config, returnTo);
    },
    [sso]
  );

  const completeKeycloakLogin = useCallback(
    async (code: string, state: string) => {
      const config = sso ?? (await fetchAuthConfig());
      setSso(config);

      const tokens = await exchangeCodeForTokens(config, code, state);
      setStoredToken(tokens.accessToken);
      if (tokens.refreshToken) {
        localStorage.setItem(REFRESH_KEY, tokens.refreshToken);
      }
      if (tokens.idToken) {
        localStorage.setItem(ID_TOKEN_KEY, tokens.idToken);
      }

      // 用刚拿到的 Keycloak 令牌向 FastAPI 取身份信息
      // （后端验签 → 自动建档/同步 → 返回公司/部门/角色）
      const me = await apiFetch<AuthUser>("/auth/me", { timeout: 15000 });
      applyMe(me);
      return tokens.returnTo;
    },
    [sso, applyMe]
  );

  const logout = useCallback(() => {
    const isSso = localStorage.getItem(ID_TOKEN_KEY) !== null;
    const idToken = localStorage.getItem(ID_TOKEN_KEY);
    const config = sso;

    clearStoredToken();
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(ID_TOKEN_KEY);
    setUser(null);

    if (isSso && config) {
      // 走 Keycloak 单点登出，否则会立刻被 SSO 会话自动登回去
      keycloakLogout(config, idToken);
      return;
    }
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
  }, [sso]);

  const can = useCallback(
    (permission: string) => {
      const permissions = user?.permissions ?? [];
      return permissions.includes("*") || permissions.includes(permission);
    },
    [user]
  );

  const value = useMemo(
    () => ({
      user,
      hydrated,
      sso,
      login,
      register,
      unifiedLogin,
      refreshUser,
      loginWithKeycloak,
      completeKeycloakLogin,
      logout,
      can,
    }),
    [
      user,
      hydrated,
      sso,
      login,
      register,
      unifiedLogin,
      refreshUser,
      loginWithKeycloak,
      completeKeycloakLogin,
      logout,
      can,
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}

/** 角色 → 中文名（后端已下发 role_label，这里只是兜底）。 */
export function roleLabel(role?: string): string {
  switch (role) {
    case "company_admin":
      return "企业管理员";
    case "kb_admin":
      return "知识库管理员";
    case "dept_manager":
    case "manager":
      return "部门负责人";
    case "employee":
    case "editor":
    case "user":
      return "普通员工";
    case "viewer":
      return "只读成员";
    case "admin":
      return "平台管理员";
    default:
      return "成员";
  }
}
