import {
  getStreamApiBase,
  ApiError,
  getStoredToken,
  clearStoredToken,
} from "@/lib/api/client";
import { normalizeDocument } from "@/lib/api/normalize";
import type { Document } from "@/lib/types";

export interface UploadProgress {
  fileName: string;
  progress: number;
}

export async function uploadDocuments(
  files: File[],
  options?: {
    collectionId?: string | null;
    onProgress?: (progress: UploadProgress) => void;
    signal?: AbortSignal;
  }
): Promise<Document[]> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  if (options?.collectionId) {
    formData.append("collection_id", options.collectionId);
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    // 直连 FastAPI 后端（与 SSE 同理）：后端要同步完成 解析→分块→嵌入
    // 整条流水线，大文件可能要几分钟；走 Next.js /api 代理会被开发
    // 服务器的代理超时中途掐断，表现为莫名其妙的红色失败提示。
    const apiBase = getStreamApiBase();
    xhr.open("POST", `${apiBase}/upload`);
    xhr.responseType = "json";
    xhr.timeout = 900000; // 15 minutes

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
