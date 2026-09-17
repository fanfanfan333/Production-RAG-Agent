/**
 * Keycloak OIDC 登录（Authorization Code + PKCE）.
 *
 * 为什么用 PKCE 而不是"后端代持密钥"：
 *   前端是公共客户端（浏览器），任何写进前端代码的 secret 都等于公开。
 *   PKCE 让每次登录带一个一次性的 code_verifier，授权码即使被截获也无法
 *   在没有 verifier 的情况下换成令牌 —— 这是 OIDC 对公共客户端的标准做法。
 *
 * 身份链路（与架构图一致）：
 *   用户 → 本模块跳转 Keycloak → 登录 → 回调带 code → 换 access_token(JWT)
 *        → 存成与本地登录同一个 rag_token → FastAPI 验签 → 身份信息
 *        (tenant_id / department / roles) → Permission Layer → Retriever
 */

import { API_BASE } from "@/lib/api/client";

export interface KeycloakPublicConfig {
  enabled: boolean;
  url: string;
  realm: string;
  client_id: string;
  issuer: string;
  auth_endpoint: string;
  token_endpoint: string;
  logout_endpoint: string;
  local_login_enabled: boolean;
}

const VERIFIER_KEY = "rag_pkce_verifier";
const STATE_KEY = "rag_pkce_state";
const RETURN_KEY = "rag_pkce_return";

/** 读取服务端下发的公开认证配置（登录页渲染前调用）。 */
export async function fetchAuthConfig(): Promise<KeycloakPublicConfig> {
  const res = await fetch(`${API_BASE}/auth/config`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error("无法获取身份认证配置");
  }
  return (await res.json()) as KeycloakPublicConfig;
}

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomString(byteLength = 32): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

async function sha256Base64Url(input: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(input)
  );
  return base64UrlEncode(new Uint8Array(digest));
}

/** 跳转到 Keycloak 登录页（会离开当前页面）。 */
export async function startKeycloakLogin(
  config: KeycloakPublicConfig,
  returnTo = "/dashboard"
): Promise<void> {
  const verifier = randomString(48);
  const state = randomString(16);
  const challenge = await sha256Base64Url(verifier);

  sessionStorage.setItem(VERIFIER_KEY, verifier);
  sessionStorage.setItem(STATE_KEY, state);
  sessionStorage.setItem(RETURN_KEY, returnTo);

  const redirectUri = `${window.location.origin}/auth/callback`;
  const params = new URLSearchParams({
    client_id: config.client_id,
    redirect_uri: redirectUri,
    response_type: "code",
    scope: "openid profile email",
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });

  window.location.href = `${config.auth_endpoint}?${params.toString()}`;
}

export interface KeycloakTokens {
  accessToken: string;
  refreshToken: string | null;
  idToken: string | null;
  expiresIn: number;
}

/** 用回调里的 code 换取令牌（PKCE 校验由 Keycloak 侧完成）。 */
export async function exchangeCodeForTokens(
  config: KeycloakPublicConfig,
  code: string,
  stateFromUrl: string
): Promise<KeycloakTokens & { returnTo: string }> {
  const verifier = sessionStorage.getItem(VERIFIER_KEY);
  const expectedState = sessionStorage.getItem(STATE_KEY);
  const returnTo = sessionStorage.getItem(RETURN_KEY) || "/dashboard";

  if (!verifier) {
    throw new Error("登录会话已失效（找不到 PKCE 校验串），请重新登录");
  }
  if (!expectedState || stateFromUrl !== expectedState) {
    throw new Error("登录状态校验失败（state 不匹配），请重新登录");
  }

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: config.client_id,
    code,
    redirect_uri: `${window.location.origin}/auth/callback`,
    code_verifier: verifier,
  });

  const res = await fetch(config.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok || !data.access_token) {
    const detail = String(
      data.error_description ?? data.error ?? "换取令牌失败"
    );
    throw new Error(detail);
  }

  sessionStorage.removeItem(VERIFIER_KEY);
  sessionStorage.removeItem(STATE_KEY);
  sessionStorage.removeItem(RETURN_KEY);

  return {
    accessToken: String(data.access_token),
    refreshToken: data.refresh_token ? String(data.refresh_token) : null,
    idToken: data.id_token ? String(data.id_token) : null,
    expiresIn: Number(data.expires_in ?? 1800),
    returnTo,
  };
}

/** 静默续期：access token 快过期时用 refresh token 换新的（不打扰用户）。 */
export async function refreshKeycloakToken(
  config: KeycloakPublicConfig,
  refreshToken: string
): Promise<KeycloakTokens> {
  const body = new URLSearchParams({
    grant_type: "refresh_token",
    client_id: config.client_id,
    refresh_token: refreshToken,
  });

  const res = await fetch(config.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok || !data.access_token) {
    throw new Error(
      String(data.error_description ?? data.error ?? "登录已过期，请重新登录")
    );
  }

  return {
    accessToken: String(data.access_token),
    refreshToken: data.refresh_token
      ? String(data.refresh_token)
      : refreshToken,
    idToken: data.id_token ? String(data.id_token) : null,
    expiresIn: Number(data.expires_in ?? 1800),
  };
}

/** 单点登出（清掉 Keycloak 侧的 SSO 会话，避免"退出后又被自动登回"）。 */
export function keycloakLogout(
  config: KeycloakPublicConfig,
  idToken: string | null
): void {
  const params = new URLSearchParams({
    post_logout_redirect_uri: `${window.location.origin}/login`,
  });
  if (idToken) params.set("id_token_hint", idToken);
  window.location.href = `${config.logout_endpoint}?${params.toString()}`;
}
