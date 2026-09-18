import type {
  AccessLevel,
  AnswerStatusInfo,
  ChatPipeline,
  CitationCheckInfo,
  CitationVerdict,
  Collection,
  DashboardStats,
  Document,
  DocumentStatus,
  EvidenceInfo,
  EvidenceSpan,
  GeneratedDocumentInfo,
  HealthInfo,
  MultimodalInfo,
  QuerySource,
} from "@/lib/types";
import type { OutputGuardInfo } from "@/lib/api/query";

function pick<T>(obj: Record<string, unknown>, keys: string[]): T | undefined {
  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) {
      return obj[key] as T;
    }
  }
  return undefined;
}

function normalizeStatus(raw: unknown): DocumentStatus {
  const value = String(raw ?? "processing").toLowerCase();
  if (value === "indexed" || value === "completed" || value === "ready") {
    return "indexed";
  }
  if (value === "already_exists") {
    return "already_exists";
  }
  if (value === "failed" || value === "error") return "failed";
  return "processing";
}

function inferType(name: string, rawType?: unknown): string {
  if (rawType) return String(rawType).toUpperCase();
  const ext = name.split(".").pop()?.toLowerCase();
  const map: Record<string, string> = {
    pdf: "PDF",
    md: "Markdown",
    txt: "文本",
    docx: "文档",
    xlsx: "表格",
    pptx: "演示文稿",
  };
  return map[ext ?? ""] ?? "文档";
}

export function normalizeDocument(raw: Record<string, unknown>): Document {
  const name = String(
    pick<string>(raw, ["name", "filename", "file_name", "title"]) ?? "未命名"
  );
  const uploadedRaw = pick<string>(raw, [
    "uploaded_at",
    "uploadedAt",
    "created_at",
    "createdAt",
  ]);

  const rawLevel = String(
    pick<string>(raw, ["access_level", "accessLevel"]) ?? "private"
  ).toLowerCase();
  const accessLevel: AccessLevel =
    rawLevel === "department" || rawLevel === "tenant" ? rawLevel : "private";

  return {
    id: String(pick<string>(raw, ["id", "document_id"]) ?? ""),
    name,
    type: inferType(name, pick(raw, ["type", "file_type", "fileType"])),
    size: Number(pick<number>(raw, ["size", "size_bytes", "file_size", "file_size_bytes"]) ?? 0),
    chunks: Number(
      pick<number>(raw, ["chunks", "chunk_count", "chunks_count"]) ?? 0
    ),
    status: normalizeStatus(pick(raw, ["status", "processing_status"])),
    uploadedAt: uploadedRaw ? new Date(uploadedRaw) : new Date(),
    collectionId: pick<string>(raw, ["collection_id", "collectionId"]),
    error: pick<string>(raw, ["error", "error_message", "message"]),

    // ── 三层知识库：层级标注（后端已给中文，缺失时本地兜底）────────────────
    accessLevel,
    accessLabel: String(
      pick<string>(raw, ["access_label", "accessLabel"]) ??
        (accessLevel === "tenant" ? "公司" : accessLevel === "department" ? "部门" : "个人")
    ),
    tenantId: pick<string>(raw, ["tenant_id", "tenantId"]),
    departmentId: (pick<string>(raw, ["department_id", "departmentId"]) ?? null) as
      | string
      | null,
    ownerUsername: (pick<string>(raw, ["owner_username", "ownerUsername"]) ??
      null) as string | null,

    // ── 权限能力 ────────────────────────────────────────────────────────────
    isOwner: Boolean(pick<boolean>(raw, ["is_owner", "isOwner"]) ?? false),
    canDelete: Boolean(pick<boolean>(raw, ["can_delete", "canDelete"]) ?? false),
    deleteDeniedReason: String(
      pick<string>(raw, ["delete_denied_reason", "deleteDeniedReason"]) ?? ""
    ),
    canRequestDelete: Boolean(
      pick<boolean>(raw, ["can_request_delete", "canRequestDelete"]) ?? false
    ),
    canPublishDepartment: Boolean(
      pick<boolean>(raw, ["can_publish_department", "canPublishDepartment"]) ?? false
    ),
    canPublishCompany: Boolean(
      pick<boolean>(raw, ["can_publish_company", "canPublishCompany"]) ?? false
    ),
    canRequestDepartment: Boolean(
      pick<boolean>(raw, ["can_request_department", "canRequestDepartment"]) ?? false
    ),
    canRequestCompany: Boolean(
      pick<boolean>(raw, ["can_request_company", "canRequestCompany"]) ?? false
    ),
    needsShareRequest: Boolean(
      pick<boolean>(raw, ["needs_share_request", "needsShareRequest"]) ?? false
    ),
    canTransferDepartment: Boolean(
      pick<boolean>(raw, ["can_transfer_department", "canTransferDepartment"]) ?? false
    ),
    transferDeniedReason: String(
      pick<string>(raw, ["transfer_denied_reason", "transferDeniedReason"]) ?? ""
    ),
    publishDeniedReason: String(
      pick<string>(raw, ["publish_denied_reason", "publishDeniedReason"]) ?? ""
    ),
    pendingShareRequest: Boolean(
      pick<boolean>(raw, ["pending_share_request", "pendingShareRequest"]) ?? false
    ),

    // ── 异步入库进度 ────────────────────────────────────────────────────────
    currentStage: (pick<string>(raw, ["current_stage", "currentStage"]) ??
      null) as string | null,
    progress: normalizeProgress(raw),

    // ── 图片（图片是独立检索对象，列表里要能看见）────────────────────────────
    imageCount: toCount(pick<number>(raw, ["image_count", "imageCount"])),
    imageObjectCount: toCount(
      pick<number>(raw, ["image_object_count", "imageObjectCount"])
    ),
  };
}

/** 计数字段归一化：非数字/负数一律归 0，避免 NaN 漏进渲染。 */
function toCount(value: unknown): number {
  const n = Number(value ?? 0);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
}

/**
 * 由后端 total_chunks / embedded_chunks 算出 0–100 的百分比。
 *
 * 还没分块（total 缺失或为 0）时返回 null —— 前端据此显示"解析中…"而不是
 * 一个永远停在 0% 的进度条，后者比不给进度更让人以为卡死。
 */
function normalizeProgress(raw: Record<string, unknown>): number | null {
  const total = Number(
    pick<number>(raw, ["total_chunks", "totalChunks"]) ?? 0
  );
  if (!Number.isFinite(total) || total <= 0) return null;
  const embedded = Number(
    pick<number>(raw, ["embedded_chunks", "embeddedChunks"]) ?? 0
  );
  const done = Number.isFinite(embedded) ? embedded : 0;
  return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
}
export function normalizeDocumentsPayload(raw: unknown): {
  documents: Document[];
  stats: DashboardStats;
} {
  if (Array.isArray(raw)) {
    const documents = raw.map((item) =>
      normalizeDocument(item as Record<string, unknown>)
    );
    return {
      documents,
      stats: deriveStats(documents),
    };
  }

  const obj = (raw ?? {}) as Record<string, unknown>;
  const list =
    (pick<unknown[]>(obj, ["documents", "items", "results"]) ?? []).map(
      (item) => normalizeDocument(item as Record<string, unknown>)
    );

  const stats: DashboardStats = {
    totalDocuments: Number(
      pick<number>(obj, [
        "total_documents",
        "totalDocuments",
        "total",
        "count",
      ]) ?? list.length
    ),
    totalChunks: Number(
      pick<number>(obj, ["total_chunks", "totalChunks"]) ??
        list.reduce((sum, doc) => sum + doc.chunks, 0)
    ),
    storageUsed: Number(
      pick<number>(obj, [
        "storage_used",
        "storageUsed",
        "storage_used_bytes",
      ]) ?? list.reduce((sum, doc) => sum + doc.size, 0)
    ),
    storageLimit: Number(
      pick<number>(obj, [
        "storage_limit",
        "storageLimit",
        "storage_limit_bytes",
      ]) ?? 1024 * 1024 * 1024
    ),
  };

  return { documents: list, stats };
}

function deriveStats(documents: Document[]): DashboardStats {
  return {
    totalDocuments: documents.length,
    totalChunks: documents.reduce((sum, doc) => sum + doc.chunks, 0),
    storageUsed: documents.reduce((sum, doc) => sum + doc.size, 0),
    storageLimit: 1024 * 1024 * 1024,
  };
}

export function normalizeCollection(raw: Record<string, unknown>): Collection {
  const createdRaw = pick<string>(raw, ["created_at", "createdAt"]);
  return {
    id: String(pick<string>(raw, ["id", "collection_id"]) ?? ""),
    name: String(pick<string>(raw, ["name", "title"]) ?? "未命名"),
    documentCount: Number(
      pick<number>(raw, [
        "document_count",
        "documentCount",
        "documents_count",
        "count",
      ]) ?? 0
    ),
    createdAt: createdRaw ? new Date(createdRaw) : new Date(),
  };
}

export function normalizeCollectionsPayload(raw: unknown): Collection[] {
  if (Array.isArray(raw)) {
    return raw.map((item) =>
      normalizeCollection(item as Record<string, unknown>)
    );
  }
  const obj = (raw ?? {}) as Record<string, unknown>;
  const list = pick<unknown[]>(obj, ["collections", "items", "results"]) ?? [];
  return list.map((item) =>
    normalizeCollection(item as Record<string, unknown>)
  );
}

function normalizeServiceStatus(raw: unknown): string {
  if (typeof raw === "boolean") return raw ? "connected" : "disconnected";
  if (typeof raw === "object" && raw !== null) {
    const obj = raw as Record<string, unknown>;
    return String(
      pick(obj, ["status", "state"]) ??
        (pick<boolean>(obj, ["connected", "healthy"]) ? "connected" : "unknown")
    );
  }
  return String(raw ?? "unknown");
}

export function normalizeHealth(raw: unknown): HealthInfo {
  const obj = (raw ?? {}) as Record<string, unknown>;
  const services = (pick<Record<string, unknown>>(obj, ["services"]) ??
    {}) as Record<string, unknown>;

  return {
    status: String(pick(obj, ["status", "health"]) ?? "unknown"),
    ollama: normalizeServiceStatus(
      pick(obj, ["ollama"]) ?? services.ollama
    ),
    qdrant: normalizeServiceStatus(
      pick(obj, ["qdrant"]) ?? services.qdrant
    ),
    postgres: normalizeServiceStatus(
      pick(obj, ["postgres"]) ?? services.postgres ?? services.database
    ),
    backend: normalizeServiceStatus(
      pick(obj, ["backend"]) ?? services.backend ?? obj.status
    ),
    environment: String(
      pick(obj, ["environment", "env"]) ?? process.env.NODE_ENV ?? "development"
    ),
    version: pick<string>(obj, ["version"]),
  };
}

export function normalizeSources(raw: unknown): QuerySource[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    const obj = item as Record<string, unknown>;
    return {
      documentId: pick<string>(obj, ["document_id", "documentId", "id"]),
      documentName: pick<string>(obj, [
        "document_name",
        "documentName",
        "filename",
        "name",
        "title",
      ]),
      chunkText: pick<string>(obj, [
        "chunk_text",
        "chunkText",
        "text",
        "content",
        "snippet",
        "text_snippet",
      ]),
      page: pick<number>(obj, ["page", "page_number"]),
      pages: pick<number>(obj, ["pages", "page_count"]),
      score: pick<number>(obj, ["score", "relevance"]),
      // ── 内容类型与图片信息（部分3 / 部分5）─────────────────────────────
      contentType: pick<string>(obj, ["content_type", "contentType"]),
      imageId: pick<string>(obj, ["image_id", "imageId"]),
      imagePath: pick<string>(obj, ["image_path", "imagePath"]),
      imageUrl: pick<string>(obj, ["image_url", "imageUrl"]),
      imageCaption: pick<string>(obj, ["image_caption", "imageCaption"]),
      vision: pick<string>(obj, ["vision"]),
      // 图片分类结论（table/chart/diagram/screenshot/photo）：
      // 前端据此在引用卡片上显示"表格 / 图表 / 流程图"徽标。
      imageType: pick<string>(obj, ["image_type", "imageType"]),
      // ── 位置信息（细粒度引用）─────────────────────────────────────────
      // 行号来自后端 chunker 写入 Qdrant payload 的 line_start/line_end；
      // location 是后端算好的"一句话溯源"，前端直接展示。
      lineStart: pick<number>(obj, ["line_start", "lineStart"]),
      lineEnd: pick<number>(obj, ["line_end", "lineEnd"]),
      lineSpan: pick<string>(obj, ["line_span", "lineSpan"]),
      location: pick<string>(obj, ["location"]),
    };
  });
}

// ── 回答依据快照（sources 之外的元数据）─────────────────────────────────────
//
// 同一批后端结构会从**两条路**到达前端：
//   1. SSE 实时事件（lib/api/query.ts）—— 回答刚生成时
//   2. 历史回放（lib/api/conversations.ts 读 messages.meta）—— 重新打开会话时
// 以前只有 (1) 有归一化代码，(2) 直接把字段丢了，于是"切窗口回来引用来源
// 就没了"。这里统一实现，两条路共用，避免两份映射各自漂移。

function numArray(raw: unknown): number[] {
  return Array.isArray(raw)
    ? (raw.filter((v) => typeof v === "number") as number[])
    : [];
}

function strArray(raw: unknown): string[] {
  return Array.isArray(raw) ? (raw as unknown[]).map(String) : [];
}

function isNonEmptyObject(raw: unknown): boolean {
  return (
    typeof raw === "object" &&
    raw !== null &&
    Object.keys(raw as Record<string, unknown>).length > 0
  );
}

/**
 * 命中句归一化（含历史回放）.
 *
 * 只保留"能定位"的条目：偏移必须是整数、文本必须非空。渲染端仍会做一次
 * 偏移↔文本的自校验，但脏数据不该流到那一层 —— 少了这一步，历史会话里
 * 的旧记录会用半截字段让高亮漂到别的句子上。
 *
 * 注意与证据门控的 normalizeEvidence（EvidenceInfo）区分：那个是"这批证据
 * 够不够答"，这个是"这条引用实际用了哪几句原文"。
 */
function normalizeEvidenceSpans(raw: unknown): EvidenceSpan[] {
  if (!Array.isArray(raw)) return [];
  return (raw as Record<string, unknown>[])
    .map((e) => ({
      start: Number(e.start ?? 0),
      end: Number(e.end ?? 0),
      text: typeof e.text === "string" ? e.text : "",
      ratio: Number(e.ratio ?? 0),
      line_start: typeof e.line_start === "number" ? e.line_start : null,
      line_end: typeof e.line_end === "number" ? e.line_end : null,
    }))
    .filter(
      (e) =>
        Number.isInteger(e.start) &&
        Number.isInteger(e.end) &&
        e.end > e.start &&
        e.text.length > 0
    );
}

export function normalizeVerdicts(raw: unknown): CitationVerdict[] {
  if (!Array.isArray(raw)) return [];
  return (raw as Record<string, unknown>[]).map((v) => ({
    index: Number(v.index ?? 0),
    sentence: String(v.sentence ?? ""),
    citation_exists: v.citation_exists === true,
    position_correct: v.position_correct === true,
    supported: v.supported === true,
    numbers_consistent: v.numbers_consistent === true,
    dates_consistent: v.dates_consistent === true,
    passed: v.passed === true,
    failed_checks: strArray(v.failed_checks),
    support_ratio: Number(v.support_ratio ?? 0),
    best_source: typeof v.best_source === "number" ? v.best_source : null,
    missing_numbers: strArray(v.missing_numbers),
    missing_dates: strArray(v.missing_dates),
    location: typeof v.location === "string" ? v.location : null,
    evidence: normalizeEvidenceSpans(v.evidence),
  }));
}

/** Citation Verifier 结论；空对象（非检索分支）返回 undefined。 */
export function normalizeCitationCheck(
  raw: unknown
): CitationCheckInfo | undefined {
  const obj = (raw ?? {}) as Record<string, unknown>;
  const verdicts = normalizeVerdicts(obj.verdicts);
  const overall = String(obj.overall ?? "");
  if (!overall && verdicts.length === 0) return undefined;
  const sanitized = obj.sanitized_answer ?? obj.sanitizedAnswer;
  return {
    overall: (overall || "no_citations") as CitationCheckInfo["overall"],
    total: Number(obj.total ?? 0),
    passed: Number(obj.passed ?? 0),
    unsupported: numArray(obj.unsupported),
    hallucinated: numArray(obj.hallucinated),
    misattributed: numArray(obj.misattributed),
    number_mismatch: numArray(obj.number_mismatch),
    date_mismatch: numArray(obj.date_mismatch),
    verdicts,
    sanitizedAnswer:
      typeof sanitized === "string" && sanitized ? sanitized : undefined,
  };
}

/** Evidence Gate 判定；未跑门控（如闲聊分支）返回 undefined。 */
export function normalizeEvidence(raw: unknown): EvidenceInfo | undefined {
  const obj = (raw ?? {}) as Record<string, unknown>;
  if (obj.passed === undefined && obj.reason === undefined) return undefined;
  return {
    passed: obj.passed !== false,
    reason: String(obj.reason ?? ""),
    confidence: Number(obj.confidence ?? 0),
    evidence_count: Number(obj.evidence_count ?? 0),
    top_score: Number(obj.top_score ?? 0),
    coverage: Number(obj.coverage ?? 0),
    failed_signals: strArray(obj.failed_signals),
  };
}

/**
 * Output Guard 审计信号.
 *
 * 后端传的是**数组**（命中了哪些片段），界面要的是**条数** —— 以前这个
 * 换算只写在 SSE 分支里，历史回放拿不到。空审计（未净化）返回 undefined，
 * 避免给一条完全合规的回答挂上无意义的徽标。
 */
export function normalizeOutputGuard(
  raw: unknown
): OutputGuardInfo | undefined {
  const obj = (raw ?? {}) as Record<string, unknown>;
  const citations_removed = numArray(obj.citations_removed);
  const leaked_phrases = strArray(obj.leaked_phrases).length;
  const hallucination_phrases = strArray(obj.hallucination_phrases).length;
  const tool_attempt_phrases = strArray(obj.tool_attempt_phrases).length;
  const changed = obj.changed === true;
  if (
    !changed &&
    !citations_removed.length &&
    !leaked_phrases &&
    !hallucination_phrases &&
    !tool_attempt_phrases
  ) {
    return undefined;
  }
  const sanitized = obj.sanitized_answer ?? obj.sanitizedAnswer;
  return {
    changed,
    citations_removed,
    leaked_phrases,
    hallucination_phrases,
    tool_attempt_phrases,
    sanitized_answer:
      typeof sanitized === "string" && sanitized ? sanitized : undefined,
  };
}

/** 图文分流统计；缺 image_count 视为未跑该节点。 */
export function normalizeMultimodal(raw: unknown): MultimodalInfo | undefined {
  const obj = (raw ?? {}) as Record<string, unknown>;
  if (obj.image_count === undefined) return undefined;
  return {
    image_count: Number(obj.image_count ?? 0),
    vision_used: Number(obj.vision_used ?? 0),
    vision_available: obj.vision_available === true,
  };
}

/** Document Agent 产物（下载卡片）；无 filename 视为未生成。 */
export function normalizeGeneratedDocument(
  raw: unknown
): GeneratedDocumentInfo | undefined {
  const obj = (raw ?? {}) as Record<string, unknown>;
  if (!obj.filename) return undefined;
  return {
    filename: String(obj.filename),
    title: String(obj.title ?? ""),
    download_url: String(
      obj.download_url ?? obj.downloadUrl ?? ""
    ),
    section_count: Number(obj.section_count ?? 0),
    table_count: Number(obj.table_count ?? 0),
    image_count: Number(obj.image_count ?? 0),
    char_count: Number(obj.char_count ?? 0),
    size_bytes: Number(obj.size_bytes ?? 0),
    error: typeof obj.error === "string" ? obj.error : null,
  };
}

/** 一条回答的完整依据快照（messages.meta）→ 界面所需的各个字段。 */
export function normalizeTurnMeta(raw: unknown): {
  intent?: ChatPipeline;
  sources?: QuerySource[];
  citationCheck?: CitationCheckInfo;
  evidence?: EvidenceInfo;
  multimodal?: MultimodalInfo;
  outputGuard?: OutputGuardInfo;
  document?: GeneratedDocumentInfo;
  answerStatus?: AnswerStatusInfo;
} {
  if (!isNonEmptyObject(raw)) return {};
  const meta = raw as Record<string, unknown>;
  const intent = String(meta.intent ?? "");
  const sources = normalizeSources(meta.sources);
  return {
    intent: intent ? (intent as ChatPipeline) : undefined,
    sources: sources.length ? sources : undefined,
    citationCheck: normalizeCitationCheck(meta.citation_check),
    evidence: normalizeEvidence(meta.evidence),
    multimodal: normalizeMultimodal(meta.multimodal),
    outputGuard: normalizeOutputGuard(meta.output_guard),
    document: normalizeGeneratedDocument(meta.document),
    answerStatus: normalizeAnswerStatus(meta.answer_status),
  };
}

/**
 * 答复性质快照（拒答 / 正常）。
 *
 * 老数据（升级前落库的回答）没有这个键 → 返回 undefined，前端按"正常回答"
 * 渲染，行为与升级前一致，不需要回填历史。
 */
export function normalizeAnswerStatus(raw: unknown): AnswerStatusInfo | undefined {
  if (!isNonEmptyObject(raw)) return undefined;
  const meta = raw as Record<string, unknown>;
  if (meta.refused === undefined && meta.sources_used === undefined) return undefined;
  const refused = Boolean(meta.refused ?? false);
  return {
    refused,
    sourcesUsed: Boolean(meta.sources_used ?? !refused),
    note: meta.note ? String(meta.note) : undefined,
  };
}
