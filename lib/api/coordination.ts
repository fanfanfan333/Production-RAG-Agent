import { apiFetch } from "@/lib/api/client";

export type CoordinationTask = {
  id: string;
  status: string;
  request_text: string;
  execution_id?: string | null;
  error?: string | null;
  reconcile_error?: string;
  stop_requested?: boolean;
  abort_detail?: string;
  result?: { summary?: string; subtasks?: Array<Record<string, unknown>> };
};

function headers(token: string): Record<string, string> {
  return { "Content-Type": "application/json", ...(token ? { "X-ForgeFlow-Token": token } : {}) };
}

export type ForgeFlowIdentity = { tenant_id: string; user_id: string };

export async function getForgeFlowLink(): Promise<ForgeFlowIdentity | null> {
  return (await apiFetch<{ identity: ForgeFlowIdentity | null }>("/agent/tasks/identity")).identity;
}

export async function linkForgeFlow(token: string): Promise<ForgeFlowIdentity> {
  return (await apiFetch<{ identity: ForgeFlowIdentity }>("/agent/tasks/identity", { method: "PUT", headers: headers(token) })).identity;
}

export async function unlinkForgeFlow(): Promise<void> {
  await apiFetch("/agent/tasks/identity", { method: "DELETE" });
}

export async function submitCoordinationTask(query: string, kind: string, token: string, key: string): Promise<CoordinationTask> {
  const result = await apiFetch<{ task: CoordinationTask }>("/agent/tasks", {
    method: "POST", headers: headers(token),
    body: JSON.stringify({ query, kind, idempotency_key: key }),
  });
  return result.task;
}

export function getCoordinationTask(id: string, token: string): Promise<CoordinationTask> {
  return apiFetch(`/agent/tasks/${encodeURIComponent(id)}`, { headers: headers(token) });
}

export async function listCoordinationTasks(): Promise<CoordinationTask[]> {
  return (await apiFetch<{ items: CoordinationTask[] }>("/agent/tasks")).items;
}

export async function stopCoordinationTask(id: string, token: string): Promise<CoordinationTask> {
  const result = await apiFetch<{ task: CoordinationTask }>(`/agent/tasks/${encodeURIComponent(id)}/stop`, {
    method: "POST", headers: headers(token),
  });
  return result.task;
}

export async function decideCoordinationTask(id: string, token: string, decision: "approve" | "reject"): Promise<CoordinationTask> {
  const result = await apiFetch<{ task: CoordinationTask }>(`/agent/tasks/${encodeURIComponent(id)}/approval`, {
    method: "POST", headers: headers(token), body: JSON.stringify({ decision }),
  });
  return result.task;
}
