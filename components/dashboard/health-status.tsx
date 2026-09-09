"use client";

import { motion } from "framer-motion";
import { RefreshCw } from "lucide-react";
import { useHealth } from "@/lib/hooks/use-health";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export function HealthStatus() {
  const { health, loading, error, refetch } = useHealth();

  const services = [
    { name: "后端 API", key: "backend", value: health?.backend },
    { name: "Ollama", key: "ollama", value: health?.ollama },
    { name: "PostgreSQL 数据库", key: "postgres", value: health?.postgres },
    { name: "Qdrant 向量库", key: "qdrant", value: health?.qdrant },
  ];

  const isOnline = (val: string | undefined) => {
    const status = String(val ?? "unknown").toLowerCase();
    return status === "connected" || status === "ok" || status === "healthy";
  };
  const isOffline = (val: string | undefined) => {
    const status = String(val ?? "unknown").toLowerCase();
    return (
      status === "not_connected" || status === "disconnected" || status === "failed"
    );
  };

  const dotColor = (val: string | undefined) =>
    isOnline(val)
      ? "bg-emerald-500"
      : isOffline(val)
        ? "bg-rose-500"
        : "bg-amber-400";

  const statusLabel = (val: string | undefined) =>
    isOnline(val) ? "在线" : isOffline(val) ? "离线" : "未知";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.15 }}
    >
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
              系统健康
            </CardTitle>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={refetch}
            disabled={loading}
            className="size-8"
          >
            <RefreshCw className={`size-3.5 text-muted-foreground ${loading ? "animate-spin" : ""}`} />
          </Button>
        </CardHeader>
        <CardContent>
          <div className="grid divide-y divide-border/60 sm:grid-cols-2 sm:divide-y-0 md:grid-cols-4 md:divide-x">
            {services.map((service) => (
              <div
                key={service.key}
                className="flex items-center justify-between gap-2 py-2.5 md:px-4 md:first:pl-0 md:last:pr-0"
              >
                <span className="truncate text-sm text-muted-foreground">
                  {service.name}
                </span>
                <span className="flex shrink-0 items-center gap-1.5">
                  <span className={`size-1.5 rounded-full ${dotColor(service.value)}`} />
                  <span className="text-xs font-medium text-foreground">
                    {statusLabel(service.value)}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}