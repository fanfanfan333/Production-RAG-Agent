import { AppShell } from "@/components/layout/app-shell";
import { ShareRequestsPanel } from "@/components/requests/share-requests-panel";

export default function RequestsPage() {
  return (
    <AppShell activePath="/requests">
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          查看申请
        </h1>
        <p className="mt-1 text-muted-foreground">
          跟踪自己提交的共享申请是否通过；有审核权限时可直接同意或拒绝下级成员的申请
        </p>
      </div>

      <ShareRequestsPanel />
    </AppShell>
  );
}
