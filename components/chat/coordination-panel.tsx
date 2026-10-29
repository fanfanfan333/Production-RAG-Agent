"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  decideCoordinationTask, getCoordinationTask, listCoordinationTasks,
  stopCoordinationTask, submitCoordinationTask, getForgeFlowLink, linkForgeFlow, unlinkForgeFlow,
  type CoordinationTask, type ForgeFlowIdentity,
} from "@/lib/api/coordination";

const active = new Set(["queued", "running", "awaiting_approval"]);
const labels: Record<string, string> = {
  queued: "已受理", running: "执行中", awaiting_approval: "等待审批",
  completed: "已完成", partial_failure: "部分失败", failed: "失败",
  unavailable: "服务不可用", interrupted: "执行中断", stopped: "已停止",
};

export function CoordinationPanel() {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("auto");
  const [credential, setCredential] = useState("");
  const [identity, setIdentity] = useState<ForgeFlowIdentity | null>(null);
  const [tasks, setTasks] = useState<CoordinationTask[]>([]);
  const [selected, setSelected] = useState<CoordinationTask | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopFeedback, setStopFeedback] = useState("");
  const retry = useRef<{ fingerprint: string; key: string } | null>(null);
  const selectedId = useRef("");
  const refreshSequence = useRef(0);

  const loadList = useCallback(async () => {
    try { setTasks(await listCoordinationTasks()); }
    catch (err) { setError(err instanceof Error ? err.message : "任务列表加载失败"); }
  }, []);
  useEffect(() => { void loadList(); }, [loadList]);
  useEffect(() => { void getForgeFlowLink().then(setIdentity).catch(() => setError("账号关联状态获取失败")); }, []);

  async function changeLink(remove: boolean) {
    setBusy(true); setError("");
    try {
      if (remove) { await unlinkForgeFlow(); setIdentity(null); }
      else { setIdentity(await linkForgeFlow(credential.trim())); }
    } catch (err) { setError(err instanceof Error ? err.message : "账号关联操作失败"); }
    finally { setBusy(false); }
  }

  const refresh = useCallback(async (id: string) => {
    const sequence = ++refreshSequence.current;
    try {
      const task = await getCoordinationTask(id, credential);
      if (selectedId.current !== id || sequence !== refreshSequence.current) return;
      setSelected(task);
      setTasks((old) => [task, ...old.filter((item) => item.id !== task.id)]);
    } catch (err) { setError(err instanceof Error ? err.message : "任务状态获取失败"); }
  }, [credential]);
  useEffect(() => {
    if (!selected || !active.has(selected.status)) return;
    const timer = setInterval(() => { void refresh(selected.id); }, 4000);
    return () => clearInterval(timer);
  }, [selected, refresh]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = query.trim();
    if (!text) return;
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify([text, kind])));
    const fingerprint = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (!retry.current) {
      try { retry.current = JSON.parse(sessionStorage.getItem("rag_coordination_pending") || "null"); } catch { retry.current = null; }
    }
    if (!retry.current || retry.current.fingerprint !== fingerprint) {
      retry.current = { fingerprint, key: crypto.randomUUID() };
    }
    sessionStorage.setItem("rag_coordination_pending", JSON.stringify(retry.current));
    setBusy(true); setError("");
    try {
      const task = await submitCoordinationTask(text, kind, credential.trim(), retry.current.key);
      retry.current = null;
      sessionStorage.removeItem("rag_coordination_pending");
      selectedId.current = task.id;
      setStopFeedback("");
      setSelected(task);
      setTasks((old) => [task, ...old.filter((item) => item.id !== task.id)]);
      setQuery("");
    } catch (err) { setError(err instanceof Error ? err.message : "任务提交失败"); }
    finally { setBusy(false); }
  }

  async function action(decision: "stop" | "approve" | "reject") {
    if (!selected) return;
    const id = selected.id;
    setBusy(true); setError("");
    try {
      const task = decision === "stop"
        ? await stopCoordinationTask(id, credential.trim())
        : await decideCoordinationTask(id, credential.trim(), decision);
      if (selectedId.current !== id) return;
      setSelected(task);
      if (decision === "stop") setStopFeedback(task.abort_detail || "停止请求已记录，执行状态尚待确认");
      await refresh(task.id);
    } catch (err) { setError(err instanceof Error ? err.message : "操作失败"); }
    finally { setBusy(false); }
  }

  return (
    <details className="mb-5 rounded-lg border border-border p-4">
      <summary className="cursor-pointer font-semibold">协作任务：文档、代码与联网</summary>
      <form onSubmit={submit} className="mt-4 grid gap-3">
        <label className="grid gap-1">任务内容
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} required rows={3} className="rounded border border-border bg-background p-2" />
        </label>
        <label className="grid gap-1">任务类型
          <select value={kind} onChange={(event) => setKind(event.target.value)} className="rounded border border-border bg-background p-2">
            <option value="auto">自动选择</option><option value="document">文档问答</option>
            <option value="code">代码任务</option><option value="research">联网研究</option>
          </select>
        </label>
        <label className="grid gap-1">ForgeFlow 个人凭证（代码任务需要）
          <input type="password" autoComplete="off" value={credential} onChange={(event) => setCredential(event.target.value)} className="rounded border border-border bg-background p-2" />
        </label>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span>{identity ? `已关联：${identity.tenant_id} / ${identity.user_id}` : "尚未关联 ForgeFlow 账号"}</span>
          <button type="button" disabled={busy || !credential.trim()} onClick={() => void changeLink(false)} className="rounded border border-border px-2 py-1">验证并关联</button>
          {identity && <button type="button" disabled={busy} onClick={() => void changeLink(true)} className="rounded border border-border px-2 py-1">解除关联</button>}
        </div>
        <button disabled={busy} className="w-fit rounded bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50">提交任务</button>
      </form>
      {error && <p role="alert" className="mt-3 text-red-600">{error}</p>}
      <div className="mt-4 flex flex-wrap gap-2" aria-label="任务列表">
        {tasks.map((task) => <button key={task.id} type="button" onClick={() => { selectedId.current = task.id; setStopFeedback(""); setSelected(task); void refresh(task.id); }} className="rounded border border-border px-2 py-1 text-left text-sm">{task.request_text?.slice(0, 30) || "任务"} · {labels[task.status] || task.status}</button>)}
      </div>
      {selected && <div className="mt-4 space-y-3" aria-live="polite">
        <p>状态：{labels[selected.status] || selected.status}{selected.stop_requested ? "（已请求停止）" : ""}</p>
        {selected.execution_id && <p className="break-all text-sm">执行编号：{selected.execution_id}</p>}
        {(selected.error || selected.reconcile_error) && <p className="text-red-600">{selected.error || selected.reconcile_error}</p>}
        {stopFeedback && <p>停止反馈：{stopFeedback}</p>}
        {selected.result?.summary && <p>{selected.result.summary}</p>}
        {selected.result?.subtasks?.map((item, index) => <section key={index} className="rounded border border-border p-3">
          <p className="font-medium">{String(item.kind || "子任务")}：{labels[String(item.status)] || String(item.status || "未知")}</p>
          <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs">{JSON.stringify(item, null, 2)}</pre>
        </section>)}
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={() => void refresh(selected.id)} className="rounded border border-border px-3 py-1">刷新</button>
          {(active.has(selected.status) || ["interrupted", "unavailable", "failed", "partial_failure"].includes(selected.status)) && <button type="button" disabled={busy} onClick={() => void action("stop")} className="rounded border border-border px-3 py-1">请求停止</button>}
          {selected.status === "awaiting_approval" && <><button type="button" disabled={busy} onClick={() => void action("approve")} className="rounded border border-border px-3 py-1">批准</button><button type="button" disabled={busy} onClick={() => void action("reject")} className="rounded border border-border px-3 py-1">拒绝</button></>}
        </div>
      </div>}
    </details>
  );
}
