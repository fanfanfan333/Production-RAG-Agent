import { apiFetch } from "@/lib/api/client";
import { normalizeDocumentsPayload } from "@/lib/api/normalize";
import type { DashboardStats, Document, DocumentChunksResponse } from "@/lib/types";

export async function getDocuments(collectionId?: string | null): Promise<{
  documents: Document[];
  stats: DashboardStats;
}> {
  const params = new URLSearchParams();
  if (collectionId) params.set("collection_id", collectionId);
  const query = params.toString();
  const raw = await apiFetch<unknown>(
    `/documents${query ? `?${query}` : ""}`
  );
  return normalizeDocumentsPayload(raw);
}

export async function getDocumentChunks(documentId: string): Promise<DocumentChunksResponse> {
  const raw = (await apiFetch<Record<string, unknown>>(`/documents/${documentId}/chunks`)) as Record<string, unknown>;
  return {
    documentId: String(raw.document_id ?? ""),
    filename: String(raw.filename ?? ""),
    pageCount: Number(raw.page_count ?? 0),
    total: Number(raw.total ?? 0),
    chunks: (raw.chunks as Record<string, unknown>[] ?? []).map((c) => ({
      chunkIndex: Number(c.chunk_index ?? 0),
      pageNumber: Number(c.page_number ?? 1),
      text: String(c.text ?? ""),
    })),
  };
}

export async function deleteDocument(id: string): Promise<void> {
  await apiFetch<void>(`/documents/${id}`, { method: "DELETE" });
}

export async function assignDocumentToCollection(
  documentId: string,
  collectionId: string | null
): Promise<void> {
  await apiFetch(`/documents/${documentId}/collection`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection_id: collectionId }),
  });
}
