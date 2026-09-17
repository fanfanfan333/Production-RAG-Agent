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
import { useAuth } from "@/lib/context/auth-context";
import { useHealth } from "@/lib/hooks/use-health";
import { BadCaseReview } from "@/components/settings/badcase-review";
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
  const { user } = useAuth();
  // RAG 质量监控（统计口径）需要 `audit.read` 权限（admin / manager）。
  // 前端角色只有 admin / user，这里按 admin 展示；真正的权限校验以后端 403 为准。
  const isAdmin = user?.role === "admin";
  const [urlInput, setUrlInput] = useState("");

  useEffect(() => {
    setUrlInput(getApiBase());
  }, []);

  const handleSaveUrl = () => {
    if (!urlInput.trim()) return;
    localStorage.setItem("rag_backend_url", urlInput.trim());
    toast.success("API 地址已更新");
    refetch();
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

      {/*
        修改密码已迁到「个人主页」（点击右上角头像）——它是"关于我"的操作，
        放在系统设置的运行时参数里语义不搭，用户也很难找到。
      */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="size-4 text-primary" />
            账号安全
          </CardTitle>
          <CardDescription>修改登录密码</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            点击右上角头像打开个人主页，在「修改密码」中修改登录密码。
          </p>
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

      {isAdmin && (
        <>
          <div className="pt-2">
            <h2 className="text-lg font-semibold tracking-tight">RAG 质量监控</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              运行期质量指标：拒答率、引用通过率与输出净化命中率
            </p>
          </div>
          <BadCaseReview />
        </>
      )}
    </div>
  );
}
