import {
  getStreamApiBase,
  ApiError,
  getStoredToken,
  clearStoredToken,
} from "@/lib/api/client";
import { normalizeDocument } from "@/lib/api/normalize";
import type { AccessLevel, Document } from "@/lib/types";

export interface UploadProgress {
  fileName: string;
  progress: number;
}

export async function uploadDocuments(
  files: File[],
  options?: {
    collectionId?: string | null;
    /** 三层知识库：文档直接落在哪一层（不传则由后端按默认层级决定）。 */
    accessLevel?: AccessLevel;
    onProgress?: (progress: UploadProgress) => void;
    signal?: AbortSignal;
  }
): Promise<Document[]> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  if (options?.collectionId) {
    formData.append("collection_id", options.collectionId);
  }
  // 目标层级由后端按权限矩阵复核：无权限的层级会被 403 拦下并提示改用申请。
  if (options?.accessLevel) {
    formData.append("access_level", options.accessLevel);
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    // 直连 FastAPI 后端（与 SSE 同理）。POST /upload 现在只做「受理」——校验、
    // 判重、落一条 PENDING 行——就返回 202，解析/OCR/向量化在服务端后台跑
    // （进度见 GET /documents 的 current_stage）。所以这个超时只需覆盖"把字节
    // 推上去"：50 MB 上限下留 5 分钟已非常宽松，超时能更快给出可读的失败提示。
    const apiBase = getStreamApiBase();
    xhr.open("POST", `${apiBase}/upload`);
    xhr.responseType = "json";
    xhr.timeout = 300000; // 5 minutes — 上传本身，不含服务端后台入库

    const token = getStoredToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    if (options?.signal) {
      options.signal.addEventListener("abort", () => xhr.abort());
    }

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || !options?.onProgress) return;
      const progress = Math.round((event.loaded / event.total) * 100);
      options.onProgress({
        fileName: files.length === 1 ? files[0].name : `${files.length} files`,
        progress,
      });
    };

    xhr.onload = () => {
      // 会话过期：清 token 并回登录页（与 apiFetch 行为一致）
      if (xhr.status === 401) {
        clearStoredToken();
        if (
          typeof window !== "undefined" &&
          window.location.pathname !== "/login"
        ) {
          window.location.href = "/login";
        }
        reject(new ApiError(401, "登录已过期，请重新登录"));
        return;
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        const raw = xhr.response;
        if (Array.isArray(raw)) {
          resolve(
            raw.map((item) =>
              normalizeDocument(item as Record<string, unknown>)
            )
          );
          return;
        }
        const obj = (raw ?? {}) as Record<string, unknown>;
        const list =
          (obj.documents as unknown[]) ??
          (obj.results as unknown[]) ??
          (obj.items as unknown[]) ??
          (raw ? [raw] : []);
        resolve(
          list.map((item) =>
            normalizeDocument(item as Record<string, unknown>)
          )
        );
        return;
      }

      const rawResponse = xhr.response;
      let errorMsg = xhr.statusText || "Upload failed";
      if (rawResponse && typeof rawResponse === "object") {
        const obj = rawResponse as Record<string, unknown>;
        if (typeof obj.detail === "string") {
          errorMsg = obj.detail;
        } else if (Array.isArray(obj.detail)) {
          errorMsg = obj.detail.map((d: any) => d?.msg ?? String(d)).join(", ");
        } else if (typeof obj.error === "string") {
          errorMsg = obj.error;
        } else if (typeof obj.message === "string") {
          errorMsg = obj.message;
        }
      }
      reject(new ApiError(xhr.status, errorMsg));
    };

    xhr.ontimeout = () => reject(new ApiError(408, "Upload timed out (15 minute limit exceeded)"));
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    xhr.onabort = () => reject(new ApiError(0, "Upload cancelled"));
    xhr.send(formData);
  });
}
