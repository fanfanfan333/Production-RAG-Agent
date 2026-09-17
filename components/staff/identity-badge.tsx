"use client";

import { CheckCircle2, Clock, ShieldAlert, ShieldQuestion } from "lucide-react";
import type { IdentityStatus } from "@/lib/types";
import { IDENTITY_LABELS } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * 身份验证状态徽标.
 *
 * 四态与产品文案一一对应：
 *   未验证（none/rejected） → 去验证（可点击，打开弹窗）
 *   审核中（pending）       → 审核中
 *   已通过（approved）      → 已通过
 */
const META: Record<
  IdentityStatus,
  { variant: "success" | "warning" | "muted" | "destructive"; icon: typeof Clock }
> = {
  none: { variant: "muted", icon: ShieldQuestion },
  pending: { variant: "warning", icon: Clock },
  approved: { variant: "success", icon: CheckCircle2 },
  rejected: { variant: "destructive", icon: ShieldAlert },
};

const STYLES: Record<string, string> = {
  success: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  warning: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  destructive: "bg-destructive/10 text-destructive",
  muted: "bg-muted text-muted-foreground",
};

export function IdentityBadge({
  status,
  onClick,
  className,
}: {
  status: IdentityStatus;
  onClick?: () => void;
  className?: string;
}) {
  const meta = META[status] ?? META.none;
  const Icon = meta.icon;
  const clickable = Boolean(onClick) && status !== "approved";

  const content = (
    <>
      <Icon className="size-3" />
      身份验证：{IDENTITY_LABELS[status] ?? "去验证"}
    </>
  );

  if (!clickable) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-medium",
          STYLES[meta.variant],
          className
        )}
      >
        {content}
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-medium",
        "transition-opacity hover:opacity-80",
        STYLES[meta.variant],
        className
      )}
    >
      {content}
    </button>
  );
}
