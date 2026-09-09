import {
  getStreamApiBase,
  ApiError,
  getStoredToken,
  clearStoredToken,
} from "@/lib/api/client";
import { normalizeSources } from "@/lib/api/normalize";
import type { QuerySource } from "@/lib/types";

export type QueryMode =
  | "rag"
  | "doc_relations"
  | "list_documents"
  | "document_summary"
  | "general_chat"
  | "knowledge_qa";

/** Query Router 的判定结果（架构图 Query Router 节点）。 */
export interface RouteInfo {
  intent: string;
  reason: string;
}

/** Retrieval Grader 的判定结果（架构图 Retrieval Grader 节点）。 */
export interface GradeInfo {
  good: boolean;
  retry: number;
  reason: string;
}

export interface StreamQueryOptions {
  query: string;
  collectionId?: string | null;
  conversationId?: string | null;
  /** "doc_relations" enables the cross-document relation pipeline (问题1). */
  mode?: QueryMode;
  signal?: AbortSignal;
  onToken: (token: string) => void;
  onThinking?: (delta: string) => void;
  onSources: (sources: QuerySource[]) => void;
  onDone: (conversationId?: string) => void;
  onError: (error: Error) => void;
  /** 可选：Query Router 判定结果（用于展示"识别为文档总结/闲聊"等状态）。 */
  onRoute?: (info: RouteInfo) => void;
  /** 可选：Retrieval Grader 判定结果（用于展示"证据不足，正在重试"）。 */
  onGrade?: (info: GradeInfo) => void;
}

/** Map a doc_digests SSE event entry to the shared QuerySource shape. */
function normalizeDigest(raw: Record<string, unknown>): QuerySource {
  const pageCount = Number(raw.page_count ?? 0);
  return {
    documentId: raw.document_id ? String(raw.document_id) : undefined,
    documentName: raw.filename ? String(raw.filename) : "未知文档",
    chunkText: typeof raw.digest === "string" ? raw.digest : undefined,
    pages: pageCount > 0 ? pageCount : undefined,
  };
}

type SseHandler = Pick<
  StreamQueryOptions,
  "onToken" | "onThinking" | "onSources" | "onDone" | "onError" | "onRoute" | "onGrade"
>;

/** 把单个 SSE data 行分发到对应回调。 */
function dispatchSseData(data: string, handlers: SseHandler) {
  if (data === "[DONE]") {
    handlers.onDone();
    return;
  }
  try {
    const parsed = JSON.parse(data) as Record<string, unknown>;
    const type = String(parsed.type ?? parsed.event ?? "");

    // Query Router 判定结果（新增事件；旧后端不会发，忽略即可）
    if (type === "intent") {
      handlers.onRoute?.({
        intent: String(parsed.intent ?? ""),
        reason: String(parsed.reason ?? ""),
      });
      return;
    }
    // Retrieval Grader 判定结果：good=false 表示正在改写重试
    if (type === "grade") {
      handlers.onGrade?.({
        good: parsed.good === true,
        retry: Number(parsed.retry ?? 0),
        reason: String(parsed.reason ?? ""),
      });
      return;
    }
    if (type === "sources" || parsed.sources) {
      handlers.onSources(normalizeSources(parsed.sources ?? parsed.data));
      return;
    }
    // Cross-document relation analysis (问题1): per-document digests
    if (type === "doc_digests") {
      const docs = Array.isArray(parsed.documents) ? parsed.documents : [];
      handlers.onSources(
        (docs as Record<string, unknown>[]).map(normalizeDigest)
      );
      return;
    }
    if (type === "thinking_delta") {
      handlers.onThinking?.(String(parsed.content ?? ""));
      return;
    }
    if (type === "error") {
      handlers.onError(
        new Error(String(parsed.message ?? parsed.content ?? "生成失败"))
      );
      return;
    }
    if (type === "done") {
      handlers.onDone(
        parsed.conversation_id ? String(parsed.conversation_id) : undefined
      );
      return;
    }

    const token = String(
      parsed.content ??
        parsed.token ??
        parsed.text ??
        parsed.delta ??
        (type === "token" ? parsed.data : "") ??
        ""
    );
    if (token) handlers.onToken(token);
  } catch {
    if (data) handlers.onToken(data);
  }
}

export async function streamQuery(options: StreamQueryOptions): Promise<void> {
  // 直连 FastAPI 后端（绕过 Next.js 代理），确保 SSE 逐 token 到达
  const base = getStreamApiBase();
  const token = getStoredToken();

  let res: Response;
  try {
    res = await fetch(`${base}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        query: options.query,
        collection_id: options.collectionId ?? undefined,
        conversation_id: options.conversationId ?? undefined,
        mode: options.mode ?? undefined,
        stream: true,
      }),
      signal: options.signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, "无法连接服务器，请确认后端服务已启动");
  }

  if (res.status === 401) {
    clearStoredToken();
    if (typeof window !== "undefined" && window.location.pathname !== "/login") {
      window.location.href = "/login";
    }
    throw new ApiError(401, "登录已过期，请重新登录");
  }
  if (res.status === 429) {
    throw new ApiError(429, "提问太频繁了，请稍后再试");
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text || res.statusText || "请求失败");
  }

  const contentType = res.headers.get("content-type") ?? "";

  // 非流式回退：后端直接返回 JSON（理论上不会发生，保留兜底）
  if (!contentType.includes("text/event-stream")) {
    const json = (await res.json()) as Record<string, unknown>;
    if (json.sources) {
      options.onSources(normalizeSources(json.sources));
    }
    const answer = String(json.answer ?? json.response ?? json.content ?? "");
    if (answer) options.onToken(answer);
    options.onDone();
    return;
  }

  if (!res.body) throw new ApiError(500, "浏览器不支持流式读取");

  // ── 流式主路径：逐行解析 SSE ─────────────────────────────────────────────
  // 不依赖事件间空行的成组切分，任何一行到达立即处理 —— token 即到即渲染。
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  const onDoneOnce = (conversationId?: string) => {
    if (finished) return;
    finished = true;
    options.onDone(conversationId);
  };
  const handlers: SseHandler = { ...options, onDone: onDoneOnce };

  const handleLine = (rawLine: string) => {
    const line = rawLine.replace(/\r$/, "");
    if (!line.startsWith("data:")) return;
    const data = line.slice(5).trim();
    if (!data) return;
    dispatchSseData(data, handlers);
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIdx: number;
      while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 1);
        handleLine(line);
      }
    }
    if (buffer.trim()) handleLine(buffer.trim());
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      return; // 用户主动停止
    }
    throw err;
  }

  onDoneOnce();
}
