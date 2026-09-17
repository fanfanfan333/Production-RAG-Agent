import {
  ApiError,
  apiFetch,
  getStoredToken,
  getStreamApiBase,
} from "@/lib/api/client";
import { normalizeDocumentsPayload } from "@/lib/api/normalize";
import type {
  AccessLevel,
  DashboardStats,
  Document,
  DocumentChunksResponse,
} from "@/lib/types";

export async function getDocuments(
  collectionId?: string | null,
  accessLevel?: AccessLevel | "all"
): Promise<{
  documents: Document[];
  stats: DashboardStats;
}> {
  const params = new URLSearchParams();
  if (collectionId) params.set("collection_id", collectionId);
  // 三层知识库筛选（个人 / 部门 / 公司）走服务端过滤：分页才有意义 ——
  // 客户端过滤只能筛当前这一页的 20 条，会造成"翻页后数量对不上"。
  if (accessLevel && accessLevel !== "all") params.set("access_level", accessLevel);
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

/**
 * 把文档在三层知识库之间移动（个人 / 部门 / 公司）。
 *
 * 需要相应权限：部门库 → 部门负责人及以上；公司库 → 知识库管理员及以上。
 * 无权限时会抛 403 并带回中文原因（应引导用户改用「申请共享」）。
 */
export async function updateDocumentVisibility(
  id: string,
  accessLevel: "private" | "department" | "tenant"
): Promise<{
  accessLevel: string;
  accessLabel: string;
  message: string;
}> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/documents/${id}/visibility`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ access_level: accessLevel }),
    }
  );
  return {
    accessLevel: String(raw.access_level ?? accessLevel),
    accessLabel: String(raw.access_label ?? ""),
    message: String(raw.message ?? "层级已更新"),
  };
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

/**
 * 把后端返回的图片相对路径（`/documents/{id}/images/xxx.png`）拼成可直接用于
 * `<img src>` 的绝对地址。
 *
 * `<img>` 无法携带 Authorization 头，因此后端该端点额外接受 `?token=`；
 * 令牌与 API 用的是同一个短期 JWT。
 */
export function getDocumentImageUrl(imageUrl?: string | null): string | null {
  if (!imageUrl) return null;
  const base = getStreamApiBase();
  const token = getStoredToken();
  const sep = imageUrl.includes("?") ? "&" : "?";
  return `${base}${imageUrl}${token ? `${sep}token=${encodeURIComponent(token)}` : ""}`;
}

/**
 * 下载 Document Agent 生成的 Word 文档。
 *
 * 走带 Authorization 的 fetch → Blob，避免 `<a href>` 直连时 401。
 */
export async function downloadGeneratedDocument(
  downloadUrl: string,
  filename: string
): Promise<void> {
  if (!downloadUrl) throw new ApiError(400, "缺少下载地址");
  const base = getStreamApiBase();
  const token = getStoredToken();

  const res = await fetch(`${base}${downloadUrl}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) {
    throw new ApiError(res.status, "下载失败，请稍后重试");
  }

  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename || "document.docx";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}
