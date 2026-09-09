export const API_BASE = "/api";

export const TOKEN_STORAGE_KEY = "rag_token";

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setStoredToken(token: string): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearStoredToken(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
}

export function getApiBase(): string {
  if (typeof window !== "undefined") {
    const stored = localStorage.getItem("rag_backend_url");
    if (stored) return stored.replace(/\/$/, "");
  }
  return (
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    API_BASE
  );
}

/**
 * Base URL for the SSE streaming endpoint (流式专用).
 *
 * Goes DIRECTLY to the FastAPI backend instead of through the Next.js
 * `/api` rewrite, so no proxy layer can ever buffer the token stream.
 * A user-configured backend URL (localStorage) still wins.
 */
export function getStreamApiBase(): string {
  if (typeof window !== "undefined") {
    const stored = localStorage.getItem("rag_backend_url");
    if (stored) return stored.replace(/\/$/, "");
  }
  return (
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    "http://localhost:8000"
  );
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function parseErrorMessage(text: string, fallback: string): string {
  if (!text) return fallback;
  try {
    const json = JSON.parse(text) as { detail?: string | { msg?: string }[] };
    if (typeof json.detail === "string") return json.detail;
    if (Array.isArray(json.detail)) {
      // pydantic v2 prefixes validator messages with "Value error, " — strip it
      return json.detail
        .map((d) => (d?.msg ?? String(d)).replace(/^Value error,\s*/, ""))
        .join("；");
    }
  } catch {
    // plain text response
  }
  return text.length > 200 ? fallback : text;
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit & { timeout?: number }
): Promise<T> {
  const timeoutMs = init?.timeout ?? 30000; // default 30s
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const apiBase = getApiBase();
  const token = getStoredToken();

  try {
    const res = await fetch(`${apiBase}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...init?.headers,
      },
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      // Global 401 handling: expired/invalid session → back to login.
      if (res.status === 401 && typeof window !== "undefined") {
        clearStoredToken();
        const here = window.location.pathname;
        if (here !== "/login") {
          window.location.href = "/login";
        }
      }
      const text = await res.text().catch(() => "");
      throw new ApiError(
        res.status,
        parseErrorMessage(text, res.statusText || "请求失败")
      );
    }

    if (res.status === 204) return undefined as T;
    return res.json() as Promise<T>;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new ApiError(408, "请求超时");
    }
    throw err;
  }
}
