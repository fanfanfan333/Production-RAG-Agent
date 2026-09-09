import { AppShell } from "@/components/layout/app-shell";
import { DashboardContent } from "@/components/dashboard/dashboard-content";

export default function DashboardPage() {
  return (
    <AppShell activePath="/dashboard">
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          仪表盘
        </h1>
        <p className="mt-1 text-muted-foreground">
          提问、上传与管理您的企业知识库
        </p>
      </div>

      <DashboardContent />
    </AppShell>
  );
}
