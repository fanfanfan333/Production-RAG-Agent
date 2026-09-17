"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Brain,
  Building2,
  FileStack,
  Home,
  Inbox,
  MessageSquare,
  Settings,
  ShieldCheck,
} from "lucide-react";
import Link from "next/link";
import { useAuth, roleLabel } from "@/lib/context/auth-context";
import { getShareRequestSummary } from "@/lib/api/share";
import { fetchStaffSummary } from "@/lib/api/staff";
import { IdentityBadge } from "@/components/staff/identity-badge";
import { ProfileDialog } from "@/components/staff/profile-dialog";
import { cn } from "@/lib/utils";

const baseNavItems = [
  { label: "首页", href: "/dashboard", icon: Home },
  { label: "文档", href: "/documents", icon: FileStack },
  { label: "对话", href: "/chat", icon: MessageSquare },
  { label: "查看申请", href: "/requests", icon: Inbox },
];

export function AppHeader({ activePath }: { activePath: string }) {
  const { user } = useAuth();
  const [shareBadge, setShareBadge] = useState(0);
  const [staffBadge, setStaffBadge] = useState(0);
  const [profileOpen, setProfileOpen] = useState(false);

  const isPlatformAdmin =
    user?.role === "admin" || user?.permissions?.includes("*");
  const canReview =
    isPlatformAdmin || Boolean(user?.permissions?.includes("staff.review"));
  const canAdminister =
    isPlatformAdmin || Boolean(user?.permissions?.includes("staff.admin"));

  const initial = (user?.display_name || user?.username || "··")
    .slice(0, 2)
    .toUpperCase();

  /**
   * 两个角标各自刷新：共享申请（文档审核）与身份验证（人员审核）。
   * 两个业务模块通过 window 事件广播变更，无需轮询。
   */
  const loadBadges = useCallback(() => {
    if (!user) return;
    getShareRequestSummary()
      .then((summary) => setShareBadge(summary.totalBadge))
      .catch(() => setShareBadge(0));
    fetchStaffSummary()
      .then((summary) => setStaffBadge(summary.totalBadge))
      .catch(() => setStaffBadge(0));
  }, [user]);

  useEffect(() => {
    loadBadges();
    const handler = () => loadBadges();
    window.addEventListener("share-requests-changed", handler);
    window.addEventListener("staff-requests-changed", handler);
    return () => {
      window.removeEventListener("share-requests-changed", handler);
      window.removeEventListener("staff-requests-changed", handler);
    };
  }, [loadBadges, activePath]);

  const navItems = [
    ...baseNavItems,
    ...(canReview || canAdminister
      ? [{ label: "管理后台", href: "/admin", icon: ShieldCheck }]
      : []),
    { label: "设置", href: "/settings", icon: Settings },
  ];

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link href="/dashboard" className="flex items-center gap-2.5">
            <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Brain className="size-4" />
            </div>
            <span className="text-[15px] font-semibold tracking-tight">
              RAG 智能助手
            </span>
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {navItems.map((item) => {
              const isActive =
                activePath === item.href ||
                (item.href !== "/dashboard" && activePath.startsWith(item.href));
              const badge =
                item.href === "/requests"
                  ? shareBadge
                  : item.href === "/admin"
                    ? staffBadge
                    : 0;

              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className={cn(
                    "relative flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                    isActive
                      ? "bg-accent font-medium text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  <item.icon className="size-4" />
                  {item.label}
                  {badge > 0 && (
                    <span className="absolute -right-0.5 -top-0.5 flex size-4 items-center justify-center rounded-full bg-destructive text-[10px] font-medium text-destructive-foreground">
                      {badge > 9 ? "9+" : badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-2">
            {/* 未通过身份验证时，顶栏常驻一个可点击的状态入口 */}
            {user && user.identity_status && user.identity_status !== "approved" && (
              <button
                type="button"
                onClick={() => setProfileOpen(true)}
                className="hidden sm:block"
                title="查看身份验证状态"
              >
                <IdentityBadge status={user.identity_status} />
              </button>
            )}

            <button
              type="button"
              onClick={() => setProfileOpen(true)}
              title="个人主页"
              className={cn(
                "flex size-8 items-center justify-center rounded-full border border-border",
                "bg-muted text-[11px] font-medium text-muted-foreground",
                "transition-colors hover:border-primary/40 hover:text-foreground"
              )}
            >
              {initial}
            </button>

            {user && (
              <button
                type="button"
                onClick={() => setProfileOpen(true)}
                className="hidden text-right leading-tight transition-opacity hover:opacity-80 sm:block"
              >
                <p className="text-xs font-medium text-foreground">
                  {user.display_name || user.username}
                </p>
                <p className="flex items-center justify-end gap-1 text-[11px] text-muted-foreground">
                  <Building2 className="size-3" />
                  {user.company_name || user.tenant_id || "default"} ·{" "}
                  {user.role_label || roleLabel(user.role)}
                </p>
              </button>
            )}
          </div>
        </div>
      </header>

      <ProfileDialog open={profileOpen} onOpenChange={setProfileOpen} />
    </>
  );
}
