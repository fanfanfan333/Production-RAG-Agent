import { AppShell } from "@/components/layout/app-shell";
import { DocumentsList } from "@/components/documents/documents-list";

export default function DocumentsPage() {
  return (
    <AppShell activePath="/documents">
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          查看文档
        </h1>
        <p className="mt-1 text-muted-foreground">
          查看所有已上传的文档，可打开查看内容或删除
        </p>
      </div>

      <DocumentsList />
    </AppShell>
  );
}
