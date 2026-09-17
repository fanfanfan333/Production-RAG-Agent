import type { Metadata } from "next";
import { AppProvider } from "@/lib/context/app-context";
import { AuthProvider } from "@/lib/context/auth-context";
import { IdentityGate } from "@/components/staff/identity-gate";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG 智能助手",
  description: "管理文档与知识库",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="font-sans antialiased">
        <AuthProvider>
          <AppProvider>
            {children}
            {/* 未通过企业身份验证时自动弹一次（根布局不随导航重挂载） */}
            <IdentityGate />
            <Toaster richColors closeButton />
          </AppProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
