"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Building2, Loader2, ShieldCheck } from "lucide-react";
import { useAuth } from "@/lib/context/auth-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

/**
 * 企业登录 / 注册 / 企业统一身份登录（单一极简入口）.
 *
 * 三种模式共用一张卡片、**就地切换不跳页**（对应产品需求「功能 2」）：
 *
 *   login     邮箱 + 密码                  —— 本地账号（含平台管理员）
 *   register  邮箱 + 密码 + 确认密码        —— 首次使用，注册后需上级身份验证
 *   unified   企业名称 + 企业职责 + 密码     —— 点「使用企业统一身份登录」进入，
 *              用**已有账号**登入：企业名称即邮箱；企业职责必须与账号在身份验证
 *              流程中登记的职责一致，由后端 /auth/unified-login 校验。
 *
 * 取舍说明：
 *   * 「企业账号」= 公司内的**邮箱**（如 zhangsan@company.com）。邮箱大小写不敏感。
 *   * 只有注册强制邮箱格式；登录与统一身份登录都放开 —— 否则 admin 这类
 *     历史账号会被自己的格式校验锁在门外。
 *   * unified 不跳 Keycloak（产品要求用已有账号直接登入）。Keycloak（OIDC + PKCE）
 *     保留为次级兜底入口，后端确实启用 SSO 时才出现。
 *   * 组件仍用项目原有的 Button / Input / 设计变量，风格与内页一致。
 */

// 与后端 auth_service.EMAIL_PATTERN 同口径（这里只为少一次往返，后端仍会复核）。
const EMAIL_RE = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$/;

type Mode = "login" | "register" | "unified";

const SUBTITLES: Record<Mode, string> = {
  login: "使用你的企业邮箱访问知识库",
  register: "使用企业邮箱注册，注册后需通过上级身份验证",
  unified: "使用已登记的邮箱、企业职责与密码登入",
};

export default function LoginPage() {
  const router = useRouter();
  const {
    user,
    hydrated,
    login,
    register,
    unifiedLogin,
    sso,
    loginWithKeycloak,
  } = useAuth();

  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [duty, setDuty] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [ssoPending, setSsoPending] = useState(false);

  // 已登录 → 直接进仪表盘
  useEffect(() => {
    if (hydrated && user) router.replace("/dashboard");
  }, [hydrated, user, router]);

  const ssoEnabled = Boolean(sso?.enabled);
  const localLoginEnabled = sso?.local_login_enabled !== false;
  const isRegister = mode === "register";
  const isUnified = mode === "unified";

  const switchMode = (next: Mode) => {
    setMode(next);
    setError(null);
    setConfirm("");
    setPassword("");
    setDuty("");
  };

  const handleKeycloak = async () => {
    setError(null);
    setSsoPending(true);
    try {
      await loginWithKeycloak("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "企业统一身份登录不可用");
      setSsoPending(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const account = username.trim();

    if (!account || !password) {
      setError(isUnified ? "请输入企业名称、企业职责和密码" : "请输入邮箱和密码");
      return;
    }
    if (isUnified && !duty.trim()) {
      setError("请输入企业职责");
      return;
    }
    // 只有注册要求邮箱格式；登录与统一身份登录放开（admin 等历史账号不是邮箱）。
    if (isRegister && !EMAIL_RE.test(account)) {
      setError("请输入有效的企业邮箱地址，例如 zhangsan@company.com");
      return;
    }
    if (isRegister && password !== confirm) {
      setError("两次输入的密码不一致");
      return;
    }

    setSubmitting(true);
    try {
      if (isUnified) {
        await unifiedLogin(account, duty.trim(), password);
      } else if (isRegister) {
        await register(account, password);
      } else {
        await login(account, password);
      }
      // 未通过身份验证的账号会在仪表盘被「身份验证」弹窗接住
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm">
        <div className="rounded-2xl border border-border/60 bg-card px-8 py-9 shadow-sm">
          <div className="mb-8 text-center">
            <h1 className="text-[19px] font-semibold tracking-tight">
              企业 <span className="font-bold">RAG</span> 知识库
            </h1>
            <p className="mt-2 text-[13px] text-muted-foreground">
              {SUBTITLES[mode]}
            </p>
          </div>

          {localLoginEnabled ? (
            <form onSubmit={handleSubmit} noValidate className="space-y-5">
              <div className="space-y-1.5">
                <label htmlFor="username" className="text-[13px] font-medium">
                  {isUnified ? "企业名称" : "邮箱"}
                </label>
                <Input
                  id="username"
                  type="email"
                  inputMode="email"
                  autoComplete="email"
                  placeholder={isUnified ? "请输入企业邮箱" : "请输入邮箱"}
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={submitting}
                  className="h-10"
                />
              </div>

              {/* 企业统一身份登录：中间一行填**已登记的**企业职责 */}
              {isUnified && (
                <div className="space-y-1.5">
                  <label htmlFor="duty" className="text-[13px] font-medium">
                    企业职责
                  </label>
                  <Input
                    id="duty"
                    autoComplete="off"
                    placeholder="如：嵌入式软件工程师"
                    value={duty}
                    onChange={(e) => setDuty(e.target.value)}
                    disabled={submitting}
                    className="h-10"
                  />
                </div>
              )}

              <div className="space-y-1.5">
                <label htmlFor="password" className="text-[13px] font-medium">
                  密码
                </label>
                <Input
                  id="password"
                  type="password"
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  placeholder={isRegister ? "至少 8 个字符" : "输入密码"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={submitting}
                  className="h-10"
                />
              </div>

              {isRegister && (
                <div className="space-y-1.5">
                  <label htmlFor="confirm" className="text-[13px] font-medium">
                    确认密码
                  </label>
                  <Input
                    id="confirm"
                    type="password"
                    autoComplete="new-password"
                    placeholder="再次输入密码"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    disabled={submitting}
                    className="h-10"
                  />
                </div>
              )}

              {error && (
                <p className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[13px] text-destructive">
                  {error}
                </p>
              )}

              <Button
                type="submit"
                className="h-10 w-full gap-2"
                disabled={submitting}
              >
                {submitting && <Loader2 className="size-4 animate-spin" />}
                {isRegister ? "创建账号" : "登录"}
              </Button>

              <button
                type="button"
                onClick={() => switchMode(mode === "login" ? "register" : "login")}
                disabled={submitting}
                className={cn(
                  "w-full text-center text-[13px] text-muted-foreground",
                  "transition-colors hover:text-foreground"
                )}
              >
                {isUnified
                  ? "返回邮箱登录"
                  : isRegister
                    ? "已有账号？返回登录"
                    : "首次使用？创建账号"}
              </button>
            </form>
          ) : (
            <p className="text-center text-[13px] text-muted-foreground">
              本地账号登录已关闭，请使用企业统一身份登录
            </p>
          )}

          {isUnified ? (
            ssoEnabled && (
              <div className="mt-6 border-t border-border/60 pt-5">
                <button
                  type="button"
                  onClick={handleKeycloak}
                  disabled={ssoPending}
                  className={cn(
                    "flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2",
                    "text-[13px] text-muted-foreground transition-colors",
                    "hover:bg-accent hover:text-foreground",
                    ssoPending && "opacity-60"
                  )}
                >
                  {ssoPending ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <ShieldCheck className="size-3.5" />
                  )}
                  使用 Keycloak 统一认证
                </button>
                <p className="mt-1.5 flex items-center justify-center gap-1.5 text-[11px] text-muted-foreground/70">
                  <Building2 className="size-3" />
                  由 Keycloak 统一认证 · 身份域 {sso?.realm}
                </p>
              </div>
            )
          ) : localLoginEnabled ? (
            <div className="mt-6 border-t border-border/60 pt-5">
              <button
                type="button"
                onClick={() => switchMode("unified")}
                disabled={submitting}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2",
                  "text-[13px] text-muted-foreground transition-colors",
                  "hover:bg-accent hover:text-foreground"
                )}
              >
                <ShieldCheck className="size-3.5" />
                使用企业统一身份登录
              </button>
              <p className="mt-1.5 text-center text-[11px] text-muted-foreground/70">
                企业名称（邮箱）+ 企业职责 + 密码，用已有账号登入
              </p>
            </div>
          ) : (
            ssoEnabled && (
              <div className="mt-6 border-t border-border/60 pt-5">
                <button
                  type="button"
                  onClick={handleKeycloak}
                  disabled={ssoPending}
                  className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-[13px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <ShieldCheck className="size-3.5" />
                  使用企业统一身份登录
                </button>
                <p className="mt-1.5 flex items-center justify-center gap-1.5 text-[11px] text-muted-foreground/70">
                  <Building2 className="size-3" />
                  由 Keycloak 统一认证 · 身份域 {sso?.realm}
                </p>
              </div>
            )
          )}
        </div>

        <p className="mt-5 text-center text-[11px] text-muted-foreground/70">
          未通过身份验证的账号仅可提交验证申请，无法访问知识库
        </p>
      </div>
    </div>
  );
}
