import { AppShell } from "@/components/layout/app-shell";
import { SettingsPanel } from "@/components/settings/settings-panel";

export default function SettingsPage() {
  return (
    <AppShell activePath="/settings">
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          系统设置
        </h1>
        <p className="mt-1 text-muted-foreground">
          系统健康状态与服务状态
        </p>
      </div>

      <SettingsPanel />
    </AppShell>
  );
}
