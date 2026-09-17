"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle } from "lucide-react";
import { AppHeader } from "@/components/layout/app-header";
import { IdentityDialog } from "@/components/staff/identity-dialog";
import { useAuth } from "@/lib/context/auth-context";
import { ThinkingDots } from "@/components/chat/thinking-dots";

export function AppShell({
  activePath,
  children,
}: {
  activePath: string;
  children: ReactNode;
}) {
  const router = useRouter();
  const { user, hydrated, refreshUser } = useAuth();
  const [identityOpen, setIdentityOpen] = useState(false);

  // Client-side auth guard: unauthenticated visitors land on /login.
  useEffect(() => {
    if (hydrated && !user) router.replace("/login");
  }, [hydrated, user, router]);

  if (!hydrated || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <ThinkingDots />
      </div>
    );
  }

  const status = user.identity_status ?? "approved";
  const unverified = status === "none" || status === "rejected";

  return (
    <div className="min-h-screen bg-background">
      <AppHeader activePath={activePath} />

      {/*
        未通过身份验证的账号能登录、但用不了知识库业务（后端身份闸门统一返回
        403）。这里给一条常驻说明，否则用户只会看到各处"请求失败"，不知道
        该做什么 —— 直接给出唯一可行的动作。
      */}
      {unverified && (
        <div className="border-b border-amber-500/20 bg-amber-500/5">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5 text-[13px] sm:px-6 lg:px-8">
            <AlertTriangle className="size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
            <span className="text-amber-700 dark:text-amber-400">
              你的企业身份尚未通过验证，暂时无法上传文档或提问。
            </span>
            <button
              type="button"
              onClick={() => setIdentityOpen(true)}
              className="font-medium text-amber-700 underline underline-offset-2 hover:opacity-80 dark:text-amber-400"
            >
              立即完善身份信息
            </button>
          </div>
        </div>
      )}

      <main className="relative mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>

      <IdentityDialog
        open={identityOpen}
        onOpenChange={setIdentityOpen}
        onChanged={() => {
          void refreshUser();
        }}
      />
    </div>
  );
}
