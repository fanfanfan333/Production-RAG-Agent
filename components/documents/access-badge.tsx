"use client";

import { Building2, Lock, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AccessLevel } from "@/lib/types";

/**
 * 三层知识库的中文标注徽标：个人 / 部门 / 公司。
 *
 * 三层用三种可区分的色与图标（不用表情符号，保持企业界面的克制感）：
 *   个人  中性灰 + 锁      —— 私有，只有我自己看得到
 *   部门  蓝     + 人群    —— 同部门按权限访问
 *   公司  墨绿   + 建筑    —— 全公司按权限访问
 */
const TIER_STYLES: Record<
  AccessLevel,
  { label: string; icon: typeof Lock; className: string }
> = {
  private: {
    label: "个人",
    icon: Lock,
    className:
      "border-slate-300/60 bg-slate-100 text-slate-700 dark:border-slate-600/50 dark:bg-slate-800/60 dark:text-slate-300",
  },
  department: {
    label: "部门",
    icon: Users,
    className:
      "border-sky-300/60 bg-sky-50 text-sky-700 dark:border-sky-700/50 dark:bg-sky-950/50 dark:text-sky-300",
  },
  tenant: {
    label: "公司",
    icon: Building2,
    className:
      "border-emerald-300/60 bg-emerald-50 text-emerald-700 dark:border-emerald-700/50 dark:bg-emerald-950/50 dark:text-emerald-300",
  },
};

export function AccessTierBadge({
  level,
  label,
  className,
}: {
  level?: AccessLevel;
  label?: string;
  className?: string;
}) {
  const style = TIER_STYLES[level ?? "private"] ?? TIER_STYLES.private;
  const Icon = style.icon;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none",
        style.className,
        className
      )}
      title={`知识库层级：${label || style.label}知识库`}
    >
      <Icon className="size-3" />
      {label || style.label}
    </span>
  );
}

/** 层级 → 中文全称（"个人知识库" / "部门知识库" / "公司知识库"）。 */
export function tierScopeName(level?: AccessLevel): string {
  if (level === "tenant") return "公司知识库";
  if (level === "department") return "部门知识库";
  return "个人知识库";
}
