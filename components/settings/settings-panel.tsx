"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  Database,
  KeyRound,
  Loader2,
  Server,
  XCircle,
} from "lucide-react";
import { getApiBase } from "@/lib/api/client";
import { changePassword } from "@/lib/api/auth";
import { useHealth } from "@/lib/hooks/use-health";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  const isHealthy =
    normalized.includes("connected") ||
    normalized.includes("healthy") ||
    normalized.includes("ok") ||
    normalized === "true";

  const isOffline =
    normalized.includes("not_connected") ||
    normalized.includes("disconnected") ||
    normalized.includes("failed");

  const label = isHealthy
    ? "正常"
    : status === "unknown"
      ? "未知"
      : isOffline
        ? "离线"
        : status;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium",
        isHealthy
          ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
          : "bg-destructive/10 text-destructive"
      )}
    >
      {isHealthy ? (
        <CheckCircle2 className="size-3" />
      ) : (
        <XCircle className="size-3" />
      )}
      {label}
    </span>
  );
}

const services = [
  { key: "ollama", label: "Ollama", icon: Bot },
  { key: "backend", label: "后端服务", icon: Server },
  { key: "qdrant", label: "Qdrant 向量库", icon: Database },
  { key: "postgres", label: "Postgres 数据库", icon: Database },
] as const;

export function SettingsPanel() {
  const { health, loading, error, refetch } = useHealth();
  const [urlInput, setUrlInput] = useState("");

  // 修改密码表单状态
  const [oldPw, setOldPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [changingPw, setChangingPw] = useState(false);
  const [pwError, setPwError] = useState<string | null>(null);

  useEffect(() => {
    setUrlInput(getApiBase());
  }, []);

  const handleSaveUrl = () => {
    if (!urlInput.trim()) return;
    localStorage.setItem("rag_backend_url", urlInput.trim());
    toast.success("API 地址已更新");
    refetch();
  };

  const handleChangePassword = async () => {
    setPwError(null);
    if (!oldPw || !newPw || !confirmPw) {
      setPwError("请填写所有密码输入框");
      return;
    }
    if (newPw.length < 8) {
      setPwError("新密码至少需要 8 个字符");
      return;
    }
    if (newPw !== confirmPw) {
      setPwError("两次输入的新密码不一致");
      return;
    }
    if (newPw === oldPw) {
      setPwError("新密码不能与当前密码相同");
      return;
    }

    setChangingPw(true);
    try {
      await changePassword(oldPw, newPw);
      toast.success("密码修改成功");
      setOldPw("");
      setNewPw("");
      setConfirmPw("");
    } catch (err) {
      setPwError(err instanceof Error ? err.message : "修改失败，请稍后再试");
    } finally {
      setChangingPw(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          <div className="flex items-center gap-2">
            <AlertCircle className="size-4 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            className="text-xs underline"
            onClick={() => refetch()}
          >
            重试
          </button>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {services.map((service, index) => {
          const Icon = service.icon;
          const status =
            health?.[service.key as keyof typeof health]?.toString() ??
            "unknown";

          return (
            <motion.div
              key={service.key}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <div className="flex items-center gap-2">
                    <div className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <Icon className="size-4" />
                    </div>
                    <CardTitle className="text-sm font-medium">
                      {service.label}
                    </CardTitle>
                  </div>
                  {loading ? (
                    <Skeleton className="h-5 w-20" />
                  ) : (
                    <StatusBadge status={status} />
                  )}
                </CardHeader>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="size-4 text-primary" />
            账号安全
          </CardTitle>
          <CardDescription>修改当前账号的登录密码</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                当前密码
              </label>
              <Input
                type="password"
                autoComplete="current-password"
                placeholder="输入当前密码"
                value={oldPw}
                onChange={(e) => setOldPw(e.target.value)}
                disabled={changingPw}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                新密码
              </label>
              <Input
                type="password"
                autoComplete="new-password"
                placeholder="至少 8 个字符"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                disabled={changingPw}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                确认新密码
              </label>
              <Input
                type="password"
                autoComplete="new-password"
                placeholder="再次输入新密码"
                value={confirmPw}
                onChange={(e) => setConfirmPw(e.target.value)}
                disabled={changingPw}
                onKeyDown={(e) => e.key === "Enter" && handleChangePassword()}
              />
            </div>
          </div>

          {pwError && (
            <p className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-sm text-destructive">
              {pwError}
            </p>
          )}

          <Button onClick={handleChangePassword} disabled={changingPw}>
            {changingPw ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <KeyRound className="size-4" />
            )}
            修改密码
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>环境设置</CardTitle>
          <CardDescription>运行时配置与系统信息</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          {loading ? (
            <Skeleton className="h-24 w-full" />
          ) : (
            <>
              <div className="flex flex-col gap-2 border-b border-border/60 pb-3">
                <span className="text-muted-foreground font-medium">API 基础地址</span>
                <div className="flex gap-2 max-w-md">
                  <Input
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="http://localhost:8000"
                    className="font-mono text-xs"
                  />
                  <Button size="sm" onClick={handleSaveUrl}>
                    保存
                  </Button>
                </div>
              </div>
              <div className="flex justify-between border-b border-border/60 py-2">
                <span className="text-muted-foreground">运行环境</span>
                <span>{health?.environment ?? "未知"}</span>
              </div>
              <div className="flex justify-between border-b border-border/60 py-2">
                <span className="text-muted-foreground">后端状态</span>
                <StatusBadge status={health?.status ?? "unknown"} />
              </div>
              {health?.version && (
                <div className="flex justify-between py-2">
                  <span className="text-muted-foreground">版本</span>
                  <span>{health.version}</span>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
