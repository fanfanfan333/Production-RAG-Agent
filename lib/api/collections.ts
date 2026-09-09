import { apiFetch } from "@/lib/api/client";
import type { Collection } from "@/lib/types";

/**
 * Business knowledge-base collections (企业落地第一阶段).
 *
 * Backed by PostgreSQL (/kb/collections) and scoped to the logged-in user.
 * The old /collections endpoints (raw Qdrant collections) are now an
 * admin-only infrastructure API and no longer used by the UI.
 */

interface RawCollection {
  id: string;
  name: string;
  description?: string | null;
  document_count: number;
  created_at?: string | null;
}

function toCollection(raw: RawCollection): Collection {
  return {
    id: raw.id,
    name: raw.name,
    documentCount: Number(raw.document_count ?? 0),
    createdAt: raw.created_at ? new Date(raw.created_at) : new Date(),
  };
}

export async function getCollections(): Promise<Collection[]> {
  const raw = await apiFetch<{ collections: RawCollection[]; total: number }>(
    "/kb/collections"
  );
  return (raw.collections ?? []).map(toCollection);
}

export async function createCollection(name: string): Promise<Collection> {
  const raw = await apiFetch<RawCollection>("/kb/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return toCollection(raw);
}

export async function deleteCollection(id: string): Promise<void> {
  await apiFetch<{ deleted: boolean }>(`/kb/collections/${id}`, {
    method: "DELETE",
  });
}
