"use client";
import { useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/api/client";
import { Button } from "@/components/ui/button";

export function GraphIndexControl({documentId}: {documentId: string}) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const stop = useRef(true);
  useEffect(() => () => {stop.current = true;}, []);
  async function build(restart = false) {
    stop.current = false;
    setBusy(true);
    try {
      while (!stop.current) {
        const result = await apiFetch<{status: string; cursor: number; total: number; error?: string}>(`/documents/${documentId}/graph-index${restart ? "?restart=true" : ""}`, {method: "POST", timeout: 45000});
        restart = false;
        setStatus(result.error || `${result.status === "completed" ? "已完成" : "索引中"} ${result.cursor}/${result.total}`);
        if (result.status === "failed" || result.status === "completed") break;
      }
      if (stop.current) setStatus("已暂停，可继续；当前分片完成后停止");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "索引失败，可重试");
    } finally {setBusy(false);}
  }
  return <div className="max-w-48 text-xs"><Button size="sm" variant="outline" onClick={busy ? () => {stop.current = true;} : () => build()}>{busy ? "暂停关联索引" : "建立/继续关联索引"}</Button>{!busy && status.startsWith("已完成") && <Button size="sm" variant="ghost" onClick={() => build(true)}>重建</Button>}{status && <p className="mt-1 text-muted-foreground" role="status">{status}</p>}</div>;
}
