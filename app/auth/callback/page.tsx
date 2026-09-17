"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { useAuth } from "@/lib/context/auth-context";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * OIDC 回调页：Keycloak 登录成功后重定向到这里（?code=...&state=...）。
 *
 * 拿到 code 后换令牌 → 建立会话 → 回到用户原本想去的页面。
 * 关键点：这个 effect 必须防止 React 严格模式下的**双次执行** ——
 * 授权码是一次性的，第二次交换会失败并把用户丢到错误页。
 * 因此用 ref 做一次性守卫，并且失败时把错误原因如实展示出来。
 */
function CallbackInner() {
  const router = useRouter();
  const params = useSearchParams();
  const { completeKeycloakLogin } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    const code = params.get("code");
    const state = params.get("state") ?? "";
    const oauthError = params.get("error");

    if (oauthError) {
      setError(
        `${params.get("error_description") ?? oauthError}（${oauthError}）`
      );
      return;
    }
    if (!code) {
      setError("回调缺少授权码（code），请重新登录");
      return;
    }

    completeKeycloakLogin(code, state)
      .then((returnTo) => {
        setDone(true);
        router.replace(returnTo || "/dashboard");
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "登录失败，请重新尝试");
      });
  }, [params, completeKeycloakLogin, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>企业统一身份登录</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error ? (
            <>
              <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                <AlertCircle className="mt-0.5 size-4 shrink-0" />
                <span>{error}</span>
              </p>
              <Button asChild className="w-full">
                <Link href="/login">返回登录页</Link>
              </Button>
            </>
          ) : done ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <CheckCircle2 className="size-4 text-emerald-500" />
              登录成功，正在进入系统…
            </p>
          ) : (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              正在校验身份并建立会话…
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function KeycloakCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-background">
          <Loader2 className="size-5 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <CallbackInner />
    </Suspense>
  );
}
