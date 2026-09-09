"use client";

import { Brain, Home, FileStack, MessageSquare, Settings, LogOut } from "lucide-react";
import Link from "next/link";
import { useAuth } from "@/lib/context/auth-context";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "首页", href: "/dashboard", icon: Home },
  { label: "文档", href: "/documents", icon: FileStack },
  { label: "对话", href: "/chat", icon: MessageSquare },
  { label: "设置", href: "/settings", icon: Settings },
];

export function AppHeader({ activePath }: { activePath: string }) {
  const { user, logout } = useAuth();
  const initial = user?.username?.slice(0, 2).toUpperCase() ?? "··";

  return (
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

            return (
              <Link
                key={item.label}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-accent font-medium text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <item.icon className="size-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <div
            className="flex size-8 items-center justify-center rounded-full border border-border bg-muted text-[11px] font-medium text-muted-foreground"
            title={user ? `${user.username}（${user.role === "admin" ? "管理员" : "成员"}）` : undefined}
          >
            {initial}
          </div>
          {user && (
            <span className="hidden text-xs text-muted-foreground sm:inline">
              {user.username}
              {user.role === "admin" && " · 管理员"}
            </span>
          )}
          <button
            type="button"
            onClick={logout}
            title="退出登录"
            className="flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <LogOut className="size-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
