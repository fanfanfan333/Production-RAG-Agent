"use client";

import { useEffect, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/app-header";
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
  const { user, hydrated } = useAuth();

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

  return (
    <div className="min-h-screen bg-background">
      <AppHeader activePath={activePath} />

      <main className="relative mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}
