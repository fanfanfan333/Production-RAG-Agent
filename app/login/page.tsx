"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Brain, Loader2, LogIn, UserPlus } from "lucide-react";
import { useAuth } from "@/lib/context/auth-context";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export default function LoginPage() {
  const router = useRouter();
  const { user, hydrated, login, register } = useAuth();

  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Already authenticated → straight to the dashboard.
  useEffect(() => {
    if (hydrated && user) router.replace("/dashboard");
  }, [hydrated, user, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!username.trim() || !password) {
      setError("请输入用户名和密码");
      return;
    }
    if (mode === "register" && password !== confirm) {
      setError("两次输入的密码不一致");
      return;
    }

    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(username.trim(), password);
      } else {
        await register(username.trim(), password);
      }
      router.replace("/dashboard");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "操作失败，请稍后再试"
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 flex flex-col items-center gap-3 text-center">
          <div className="flex size-12 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
            <Brain className="size-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">RAG 智能助手</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              企业私有知识库问答平台
            </p>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{mode === "login" ? "登录" : "注册账号"}</CardTitle>
            <CardDescription>
              {mode === "login"
                ? "使用你的账号访问知识库"
                : "第一个注册的账号将自动成为管理员"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <label htmlFor="username" className="text-sm font-medium">
                  用户名
                </label>
                <Input
                  id="username"
                  autoComplete="username"
                  placeholder="字母、数字、点、下划线或连字符"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={submitting}
                />
              </div>
              <div className="space-y-1.5">
                <label htmlFor="password" className="text-sm font-medium">
                  密码
                </label>
                <Input
                  id="password"
                  type="password"
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  placeholder={mode === "register" ? "至少 8 个字符" : "输入密码"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={submitting}
                />
              </div>
              {mode === "register" && (
                <div className="space-y-1.5">
                  <label htmlFor="confirm" className="text-sm font-medium">
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
                  />
                </div>
              )}

              {error && (
                <p className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                  {error}
                </p>
              )}

              <Button type="submit" className="w-full" disabled={submitting}>
                {submitting ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : mode === "login" ? (
                  <LogIn className="size-4" />
                ) : (
                  <UserPlus className="size-4" />
                )}
                {mode === "login" ? "登录" : "注册"}
              </Button>
            </form>

            <button
              type="button"
              onClick={() => {
                setMode(mode === "login" ? "register" : "login");
                setError(null);
                setConfirm("");
              }}
              disabled={submitting}
              className={cn(
                "mt-4 w-full text-center text-sm text-muted-foreground",
                "transition-colors hover:text-foreground"
              )}
            >
              {mode === "login"
                ? "还没有账号？注册一个"
                : "已有账号？返回登录"}
            </button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
