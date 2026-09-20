export type DocumentStatus = "indexed" | "processing" | "failed" | "already_exists";

/**
 * 三层知识库的层级（与后端 tenancy.ACCESS_* 一一对应）。
 *   private    个人知识库 —— 仅归属人可见
 *   department 部门知识库 —— 同部门按权限访问
 *   tenant     公司知识库 —— 全公司按权限访问
 */
export type AccessLevel = "private" | "department" | "tenant";

export interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  chunks: number;
  status: DocumentStatus;
  uploadedAt: Date;
  collectionId?: string;
  error?: string;

  // ── 异步入库进度 ──────────────────────────────────────────────────────────
  // POST /upload 受理即返回，真正的解析/向量化在后台跑。这两个字段让"处理中"
  // 说得出话：是在解析内容，还是向量化到第几批。
  /** 后端 current_stage：pending / parsing / chunking / embedding / completed … */
  currentStage?: string | null;
  /** 0–100；仅当后端给出 total_chunks 时才有值。 */
  progress?: number | null;

  // ── 图片 ────────────────────────────────────────────────────────────────
  // 图片在本项目里是**独立检索对象**（命中后回显原图），所以列表里要能看出
  // 这份文档解析出了几张图、其中几张进了检索库 —— 否则用户完全感知不到
  // "图文一体的知识库"这件事发生过。
  /** 解析出的内嵌图片总数。 */
  imageCount?: number;
  /** 其中成为独立检索对象的图片块数。 */
  imageObjectCount?: number;

  // ── 三层知识库（个人 / 部门 / 公司）─────────────────────────────────────
  /** 层级标识；accessLabel 是直接可渲染的中文标注。 */
  accessLevel?: AccessLevel;
  accessLabel?: string;
  tenantId?: string;
  departmentId?: string | null;
  /** 公司展示名（后端 tenant_name，公司注册表权威源；取不到为 null）。 */
  tenantName?: string | null;
  /** 部门展示名（后端 department_name；取不到为 null）。 */
  departmentName?: string | null;
  /** 归属人用户名（共享文档会显示"由 XXX 上传"）。 */
  ownerUsername?: string | null;

  // ── 权限能力（后端下发，按钮显隐据此判断，避免"点了才 403"）───────────────
  isOwner?: boolean;
  canDelete?: boolean;
  deleteDeniedReason?: string;
  /**
   * 自己没有删除权、但看得见该文档 → 可以提交「申请删除」由上级审核。
   * 与 canDelete 互斥：能直接删就不需要申请。
   */
  canRequestDelete?: boolean;
  canPublishDepartment?: boolean;
  canPublishCompany?: boolean;
  /**
   * 申请共享能力**按目标层级**下发（与 canPublish* 对同一层级互斥）。
   *
   * 部门负责人是典型形态：可直接发部门库、只能申请公司库 —— 二者不同层，
   * 因此 canPublishDepartment 与 canRequestCompany 同时为真。界面不能再用
   * "有任一发布权 → 隐藏申请入口"这种总布尔判断，否则他提不上去。
   */
  canRequestDepartment?: boolean;
  canRequestCompany?: boolean;
  /** 是否有任何一层需要走「申请共享」。 */
  needsShareRequest?: boolean;
  /**
   * 能否把这份文档**改归到指定部门**（公司级管理者：企业管理员 / 知识库管理员
   * / 平台管理员）。与 canPublishDepartment 是两件事：后者发的是"我自己的
   * 部门"，目标部门没有选择余地；本字段对应界面上的「转为部门文档」，由操作者
   * 在「公司已有部门」清单里选定目标 —— 典型场景是 HR 把薪酬制度下沉给人力部。
   *
   * 只在文档已共享（部门库 / 公司库）时为真：个人库文档要先共享出去。
   */
  canTransferDepartment?: boolean;
  /** 不能转为部门文档时的中文原因（个人库文档会带说明）。 */
  transferDeniedReason?: string;
  publishDeniedReason?: string;
  /** 是否已有待审核的共享申请。 */
  pendingShareRequest?: boolean;
}

export interface DashboardStats {
  totalDocuments: number;
  totalChunks: number;
  storageUsed: number;
  storageLimit: number;
}

export interface Collection {
  id: string;
  name: string;
  documentCount: number;
  createdAt: Date;
}

export interface QuerySource {
  documentId?: string;
  documentName?: string;
  chunkText?: string;
  page?: number;
  /** Total pages of the source document (doc-relation analysis entries). */
  pages?: number;
  score?: number;
  /**
   * 内容类型（部分3）：`text` | `table` | `image`.
   * image 类型的引用会渲染原始图片缩略图，而不是只显示文本片段。
   */
  contentType?: "text" | "table" | "image" | string;
  /** 图片对象 ID（稳定标识，形如 {document_id}-p3-i1）。 */
  imageId?: string;
  /** 图片在文档目录下的相对路径：images/page_3_image_1.png */
  imagePath?: string;
  /** 原始图片的后端访问地址（需带 token，见 getDocumentImageUrl）。 */
  imageUrl?: string;
  /** 入库期多模态模型生成的图片说明。 */
  imageCaption?: string;
  /** 检索期 Vision 针对当前问题的看图结论（部分5/6）。 */
  vision?: string;
  /**
   * 图片类型 —— Picture Classification 的结论。
   * `table`（表格图片，已还原成 Markdown 表格）| `chart`（图表）|
   * `diagram`（流程图/结构图）| `screenshot`（截图）| `photo`（普通图片）。
   * 引用卡片据此显示类型徽标：表格 / 图表 / 流程图 / 截图 / 图片。
   */
  imageType?: ImageType | string;
  /**
   * 位置信息（细粒度引用）：被引片段在原文中的起止行号（1-based，闭区间）。
   * 旧索引没有该字段时为 undefined，引用卡片自动降级为只显示页码。
   */
  lineStart?: number;
  lineEnd?: number;
  /** 人类可读行号区间，如 "12-28"。 */
  lineSpan?: string;
  /**
   * 一句话溯源：`《年报.pdf》第 3 页，第 12-28 行`。
   * 由后端 location_label() 统一生成，前端直接展示，避免各处文案漂移。
   */
  location?: string;
}

/** 三层图片处理里的图片类型（与后端 image_understanding 保持一致）。 */
export type ImageType =
  | "table"
  | "formula"
  | "code"
  | "chart"
  | "diagram"
  | "screenshot"
  | "photo";

/**
 * 图片类型 → 展示文案。
 *
 * ⚠️ 必须与后端 `image_understanding/structured_content.IMAGE_TYPE_LABELS` 保持
 * 同一套取值。此前这里缺 `formula` / `code` 两项，导致**公式图片与代码截图在
 * 引用卡片上被统一显示成「图片」** —— 后端明明识别出了公式并走了公式引擎，
 * 界面上却完全看不出来，用户也没法区分"这段是公式识别结果"还是"普通配图"。
 */
export const IMAGE_TYPE_LABELS: Record<string, string> = {
  table: "表格",
  formula: "公式",
  code: "代码",
  chart: "图表",
  diagram: "流程图",
  screenshot: "截图",
  photo: "图片",
};

/** Document Agent（Word 生成）产物元信息（SSE `document` 事件）。 */
export interface GeneratedDocumentInfo {
  filename: string;
  title: string;
  download_url: string;
  section_count: number;
  table_count: number;
  image_count: number;
  char_count: number;
  size_bytes: number;
  error?: string | null;
}

/** multimodal_context 节点产出的图文分流统计（SSE `multimodal` 事件）。 */
export interface MultimodalInfo {
  image_count: number;
  vision_used: number;
  vision_available: boolean;
}

/** Evidence Gate 判定结果（SSE `evidence` 事件）。 */
export interface EvidenceInfo {
  passed: boolean;
  reason: string;
  confidence: number;
  evidence_count: number;
  top_score: number;
  coverage: number;
  failed_signals: string[];
}

/**
 * Citation Verifier 的单条引用结论（五项校验）。
 *
 * 对应设计稿：引用存在？ / 引用位置正确？ / 原文支持该结论？ /
 * 数字是否一致？ / 日期是否一致？
 */
export interface CitationVerdict {
  index: number;
  sentence: string;
  citation_exists: boolean;
  position_correct: boolean;
  supported: boolean;
  numbers_consistent: boolean;
  dates_consistent: boolean;
  passed: boolean;
  failed_checks: string[];
  support_ratio: number;
  best_source: number | null;
  missing_numbers: string[];
  missing_dates: string[];
  /** 一句话溯源（《x.pdf》第 3 页，第 12-28 行）。 */
  location: string | null;
  /**
   * 命中句：这条引用实际依据的几句原文（句级回标）。
   *
   * 缺失/为空表示"这条引用的依据定位不到具体句子"（旧后端、或与原句
   * 无内容交集）——前端此时保持原样展示整段，不做任何高亮。
   */
  evidence?: EvidenceSpan[];
  /**
   * 被引来源是否含**可比对的证据文本**（能否真的拿它跟句子做内容词比对）。
   *
   * `false` = 来源拿不到任何 token/片段（不透明来源、纯图片既无 OCR 也无视觉
   * 分析），此时 `supported=false` 只是"无从判断"，**不得**据此把句子标成
   * "无依据" —— 否则中文句、图片块(vision)、表格块来源的句子会被误标。
   * 前端只对 `evidence_available !== false` 的条目做"无依据句"标注。
   */
  evidence_available?: boolean;
}

/**
 * 命中句 —— 答案**实际依据的那一句原文**（Citation Verifier 的句级回标）。
 *
 * 为什么需要它：引用此前只到"切片"这一级（"第 83-105 行" + 整段原文），
 * 用户看到 23 行原文却不知道答案用的是哪几句。"引用"应该落到句 —— 卡片
 * 默认只列这几句，展开才看整段上下文。
 */
export interface EvidenceSpan {
  /** 该句在 chunkText 中的字符区间（左闭右开）。 */
  start: number;
  end: number;
  /**
   * 原句文本。
   *
   * 偏移与文本两份信息都在：前端先按偏移切，配合文本做一次自校验，
   * 对不上就退化为按文本查找 —— 历史回放的字段口径与实时流不一定完全
   * 一致，只信偏移会把高亮整体错位，而错位的高亮比没有高亮更糟。
   */
  text: string;
  /** 与答案句的内容词重合率 ∈ [0,1]（后端确定性比对，非模型打分）。 */
  ratio: number;
  /** 该句自身的行号（1-based，闭区间）；旧索引无行号时为 null。 */
  line_start: number | null;
  line_end: number | null;
}

/** Citation Verifier 整体结论（SSE `citation_check` 事件）。 */
export interface CitationCheckInfo {
  overall: "verified" | "partial" | "unsupported" | "no_citations" | "refused_by_model";
  total: number;
  passed: number;
  unsupported: number[];
  hallucinated: number[];
  misattributed: number[];
  number_mismatch: number[];
  date_mismatch: number[];
  verdicts: CitationVerdict[];
  /**
   * 净化改动了正文（移除引用标记 / 追加校验脚注）时后端回传的完整答案。
   * 流出的 token 无法撤回，前端用这份全文整段替换已渲染内容，
   * 保证「用户看到的 == 落库的 == 校验过的」。
   */
  sanitizedAnswer?: string;
}

export interface DocumentChunk {
  chunkIndex: number;
  pageNumber: number;
  text: string;
}

export interface DocumentChunksResponse {
  documentId: string;
  filename: string;
  pageCount: number;
  total: number;
  chunks: DocumentChunk[];
}

export type FeedbackRating = "up" | "down";

export interface HealthInfo {
  status: string;
  ollama: string;
  qdrant: string;
  postgres: string;
  backend: string;
  environment: string;
  version?: string;
}

/**
 * Pipeline labels surfaced in the chat UI.
 *
 * - `rag`               — legacy alias kept for back-compat with old sessions;
 *                         new code should NOT hard-code it (let the backend
 *                         LLM router decide via mode=null at the API layer).
 * - `doc_relations`     — cross-document relation analysis
 * - `list_documents`    — deterministic DB listing (no LLM)
 * - `knowledge_qa`      — Hybrid RAG (rewrite → retrieve → grade → generate)
 * - `document_summary`  — per-document content summary
 * - `general_chat`      — no retrieval at all (idle / chitchat / identity)
 * - `document_agent`    — Document Agent: 生成可下载的 Word 文档（含原始图片）
 */
export type ChatPipeline =
  | "rag"
  | "doc_relations"
  | "list_documents"
  | "knowledge_qa"
  | "document_summary"
  | "general_chat"
  | "document_agent";

import type { OutputGuardInfo } from "@/lib/api/query";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  sources?: QuerySource[];
  isStreaming?: boolean;
  error?: string;
  /** Frontend-suggested pipeline (set when the query text matches a deterministic
   *  local rule); the server may override it via SSE `intent` events. */
  mode?: ChatPipeline | null;
  /** Server-side Query Router verdict (more authoritative than `mode`).
   *  Reflects what the master graph actually executed. */
  intent?: ChatPipeline;
  /** Output Guard audit signal (问题3+问题4): citations/leaked/hallucination/tools. */
  outputGuard?: OutputGuardInfo;
  /** Document Agent 产物：生成的 Word 文档（可下载）。 */
  document?: GeneratedDocumentInfo;
  /** multivisual_context 节点的图文分流统计（图片数 / Vision 使用情况）。 */
  multimodal?: MultimodalInfo;
  /** Evidence Gate 判定（证据是否足够；不足时后端直接拒答）。 */
  evidence?: EvidenceInfo;
  /** Citation Verifier 的五项引用校验结论。 */
  citationCheck?: CitationCheckInfo;
  /** Local-only feedback state (👍/👎). */
  feedback?: FeedbackRating;
  /** Conversation id at the time this answer finished streaming. */
  conversationId?: string;
  /** 答复性质：拒答时来源不作为依据（避免"拒答 + 引用"自相矛盾）。 */
  answerStatus?: AnswerStatusInfo;
}

export interface ConversationSummary {
  id: string;
  title: string;
  messageCount: number;
  createdAt?: string;
  updatedAt?: string;
}

/**
 * 一条历史消息（GET /conversations/{id}/messages）。
 *
 * assistant 侧除正文外还带回**当时的依据快照**（messages.meta 列）：
 * 引用来源、引用校验、证据门控、输出合规、生成文档、图文分流统计。
 * 这些字段此前完全丢失，导致重新打开历史对话时"上一次提问的数据来源
 * 不见了" —— 现在按原样还原到界面上。
 */
export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
  /** 这轮回答走过的管线（服务端路由结果）。 */
  intent?: ChatPipeline;
  /** 引用来源（含图片对象：image_url / image_type / vision）。 */
  sources?: QuerySource[];
  /** 五项引用校验结论（逐条"已核验 / 存疑"）。 */
  citationCheck?: CitationCheckInfo;
  /** 证据门控判定（不足时显示"证据不足"）。 */
  evidence?: EvidenceInfo;
  /** 图文分流 + Vision 使用统计。 */
  multimodal?: MultimodalInfo;
  /** 输出合规校验审计信号。 */
  outputGuard?: OutputGuardInfo;
  /** Document Agent 产物（下载卡片）。 */
  document?: GeneratedDocumentInfo;
  /** 答复性质（拒答时来源不作为依据）。 */
  answerStatus?: AnswerStatusInfo;
}

export const quickActions = [
  {
    id: "upload",
    label: "上传文档",
    description: "拖拽或浏览选择文件",
    icon: "upload" as const,
    href: "/dashboard#upload",
  },
  {
    id: "documents",
    label: "查看文档",
    description: "打开查看全部已上传文档",
    icon: "documents" as const,
    href: "/documents",
  },
  {
    id: "requests",
    label: "查看申请",
    description: "跟踪共享申请是否通过，处理待我审核",
    icon: "requests" as const,
    href: "/requests",
  },
  {
    id: "collections",
    label: "知识库分组",
    description: "按团队或项目组织文档",
    icon: "collections" as const,
    href: "/collections",
  },
  {
    id: "settings",
    label: "系统设置",
    description: "服务与参数配置",
    icon: "settings" as const,
    href: "/settings",
  },
] as const;

export const suggestedQuestions = [
  "这些文档之间有什么关联？",
  "我的知识库里有哪些文档？",
  "总结一下我上传文件的要点。",
  "文档里的图表说明了什么？",
  "帮我把要点整理成一份 Word 文档。",
  "帮我查找关于产品需求的信息。",
];

/** Queries matching both patterns are routed to the doc-relation pipeline. */
const RELATION_HINT_RE =
  /(关联|关系|联系|相关性|关联度|异同|共同点|相同点|相似之处|重叠|互补|主题分布)/;
const COLLECTION_HINT_RE =
  /(这些|这批|这几个|这几份|各个|所有|全部|哪些|库里|库中|库内|知识库|文档库|上传)[^。？?!\n]{0,12}(文档|文件|资料|报告)|文档库|知识库|(文档|文件|资料|报告)之间/;

export function isDocRelationQuery(query: string): boolean {
  return RELATION_HINT_RE.test(query) && COLLECTION_HINT_RE.test(query);
}

/** "知识库里有哪些文档" style listing questions → deterministic DB listing.
 *  Mirrors the backend `_ASKS_DOC_LIST_RE` (backend/app/api/query.py). */
const DOC_LIST_HINT_RE =
  /(有哪些|有什么|都有哪些|都有什么|多少个?|哪些|列出|列一下|清单|列表|包含哪些)[^。？?!\n]{0,4}(文档|文件|资料|pdf)|(文档|文件|资料|pdf)(列表|清单)|上传了(哪些|什么)(文档|文件|资料)?/i;

export function isDocumentListQuery(query: string): boolean {
  return DOC_LIST_HINT_RE.test(query);
}

/** Document Agent（生成 Word 文档）意图的本地预判。
 *  Mirrors the backend `_ASKS_DOCUMENT_AGENT_RE` (intent_rules.py)。 */
const DOC_AGENT_HINT_RE =
  /(生成|制作|导出|输出|撰写|写|整理|汇总|做成|形成|创建|帮我写|帮我做|拟)[^。？?!\n]{0,12}(word|docx|文档|报告|说明书|纪要|方案|总结报告|分析报告)|(导出|下载)[^。？?!\n]{0,8}(word|docx|文档)/i;

export function isDocumentAgentQuery(query: string): boolean {
  return DOC_AGENT_HINT_RE.test(query);
}

// ── 共享申请（申请共享 / 查看申请 / 审核）─────────────────────────────────────

export type ShareRequestStatus = "pending" | "approved" | "rejected" | "cancelled";
export type ShareTargetLevel = "department" | "tenant";

/**
 * 申请意图：发布到更高层级 / 删除一份自己没有删除权的文档。
 * 两者共用同一条审批链路（部门级 → 部门负责人；公司级 → 知识库管理员）。
 */
export type ShareIntent = "publish" | "delete";

export interface ShareRequestItem {
  id: string;
  /** 删除申请被批准后文档已不存在 —— 此时为 null。 */
  documentId: string | null;
  documentName: string;
  intent: ShareIntent;
  /** "申请共享" / "申请删除" */
  intentLabel: string;
  requesterUsername: string;
  requesterDepartmentId?: string | null;
  targetLevel: ShareTargetLevel;
  /** "部门知识库" / "公司知识库" */
  targetLabel: string;
  targetDepartmentId?: string | null;
  reason?: string | null;
  status: ShareRequestStatus;
  reviewerUsername?: string | null;
  reviewComment?: string | null;
  createdAt?: string | null;
  reviewedAt?: string | null;
  requesterSeen: boolean;
  isMine: boolean;
  canReview: boolean;
}

export interface ShareRequestSummary {
  pendingForMe: number;
  myPending: number;
  myDecidedUnseen: number;
  reviewScope: "company" | "department" | null;
  canReview: boolean;
  totalBadge: number;
}

// ── 答复性质（拒答 / 正常回答）────────────────────────────────────────────────

/**
 * 后端在流结束前下发的确定性状态。
 *
 * 存在的理由：`sources` 事件在检索后立刻发出，而"是否拒答"要到证据门控
 * 才确定 —— 拒答时若不修正，界面会出现"答不出来"与"1 个引用来源"并存的
 * 自相矛盾画面。前端据此把来源标注为"未采用"。
 */
export interface AnswerStatusInfo {
  refused: boolean;
  sourcesUsed: boolean;
  note?: string;
}

// ── 企业身份验证与层级授权（企业管理后台）─────────────────────────────────────

/**
 * 身份验证状态（与后端 staff_service 的 IDENTITY_* 一一对应）。
 *
 *   none      从未提交 → 首页强制弹窗、个人主页显示「身份验证：去验证」
 *   pending   待审核   → 显示「身份验证：审核中」
 *   approved  已通过   → 显示「身份验证：已通过」，业务功能放行
 *   rejected  被拒绝   → 显示「去验证」，可修改后重新提交
 */
export type IdentityStatus = "none" | "pending" | "approved" | "rejected";

export const IDENTITY_LABELS: Record<IdentityStatus, string> = {
  none: "去验证",
  pending: "审核中",
  approved: "已通过",
  rejected: "去验证",
};

export type StaffRequestStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "cancelled";

/** 一份企业身份验证申请（含审核人与「负责人 = 职务 + 名称」）。 */
export interface StaffRequestItem {
  id: string;
  applicantUsername: string;
  applicantDisplayName?: string | null;
  /** 展示用姓名（display_name 优先，回退账号）。 */
  applicantLabel: string;
  companyName: string;
  companyId: string;
  departmentName: string;
  departmentId: string;
  /** 部门职责（申请目标职位，如「嵌入式软件工程师」）。 */
  duty: string;
  status: StaffRequestStatus;
  reviewerUsername?: string | null;
  reviewerTitle?: string | null;
  reviewerName?: string | null;
  /** 「职务 名称」，直接渲染在申请记录后方。 */
  reviewerLabel?: string | null;
  reviewComment?: string | null;
  grantedRole?: string | null;
  grantedRoleLabel?: string | null;
  createdAt?: string | null;
  reviewedAt?: string | null;
  applicantSeen: boolean;
  isMine: boolean;
  canReview: boolean;
}

/** 身份状态 + 审核角标（导航栏与个人主页共用）。 */
export interface StaffSummary {
  identityStatus: IdentityStatus;
  pendingForMe: number;
  myPending: number;
  myDecidedUnseen: number;
  canReview: boolean;
  canAdminister: boolean;
  totalBadge: number;
}

/** 可授予的角色（前端下拉框，避免"选了才 403"）。 */
export interface GrantableRole {
  value: string;
  label: string;
}

/** 个人主页所需的完整身份信息。 */
export interface StaffProfile {
  id: string;
  username: string;
  name: string;
  displayName?: string | null;
  role: string;
  roleLabel: string;
  isAdmin: boolean;
  isActive: boolean;
  companyId: string;
  /** 公司名称原文（可能是中文，界面直接显示它）。 */
  companyName: string;
  departmentId?: string | null;
  departmentName?: string | null;
  /** 部门职责 / 职务。 */
  jobTitle?: string | null;
  authSource: string;
  identityStatus: IdentityStatus;
  grantableRoles: string[];
  canReview: boolean;
  canAdminister: boolean;
  latestRequest?: StaffRequestItem | null;
  createdAt?: string | null;
}

/** 成员管理列表里的一行。 */
export interface StaffMember {
  id: string;
  username: string;
  displayName?: string | null;
  name: string;
  role: string;
  roleLabel: string;
  isAdmin: boolean;
  isActive: boolean;
  companyId: string;
  companyName: string;
  departmentId?: string | null;
  departmentName?: string | null;
  jobTitle?: string | null;
  identityStatus: IdentityStatus;
  authSource: string;
  createdAt?: string | null;
}

export interface CompanyOption {
  companyId: string;
  companyName: string;
  memberCount: number;
  /** 是否为测试公司（注册表 ``is_test``），用于列表区分「测试公司 / 普通公司」。 */
  isTest: boolean;
  /** 当前用户能否给该公司改名（平台管理员：自建或无主公司为 true）。 */
  canRename: boolean;
}

/**
 * 删除成员前的数据影响预检（后端 /staff/members/{id}/deletion-preview）。
 *
 * 数字全部来自数据库：确认弹窗里写的"会删掉什么、会留下什么"必须与真正
 * 执行的删除是同一份口径，否则就成了免责文案。
 */
export interface MemberDeletionImpact {
  member: {
    id: string;
    username: string;
    name: string;
    companyName: string;
    departmentName?: string | null;
    jobTitle?: string | null;
    role: string;
    roleLabel: string;
  };
  /** 会被永久删除的个人数据。 */
  deleted: {
    personalDocuments: number;
    conversations: number;
    messages: number;
    collections: number;
    feedback: number;
  };
  /** 会被保留的组织资产与留痕。 */
  kept: {
    sharedDocuments: number;
    staffRequests: number;
    shareRequests: number;
    badCases: number;
  };
}

/**
 * 删除公司前的数据影响预检（后端 /companies/{id}/deletion-preview）。
 *
 * 与 :interface:`MemberDeletionImpact` 同一套「将删除 / 将保留」视觉语言，但范围
 * 是**整家公司**：员工账号、三级文档（个人 / 部门 / 公司）、会话与向量全删，
 * 只留审计与审核留痕。数字全部来自数据库，与真正执行的删除同一份口径。
 */
export interface CompanyDeletionImpact {
  company: {
    id: string;
    name: string;
    isTest: boolean;
  };
  /** 会被永久删除的数据。 */
  deleted: {
    members: number;
    documents: number;
    /** 三级文档分布：个人 / 部门 / 公司。 */
    documentsPrivate: number;
    documentsDepartment: number;
    documentsTenant: number;
    conversations: number;
    messages: number;
    collections: number;
    feedback: number;
    vectors: number;
  };
  /** 会被保留的组织留痕。 */
  kept: {
    staffRequests: number;
    shareRequests: number;
    badCases: number;
    /** 本公司成员名下、归属**其它公司**的文档：只解绑归属、不删除。 */
    crossTenantDocuments: number;
  };
}

/** 层级（数值越大权限越高）——与后端 staff_service.ROLE_RANK 保持一致。 */
export const ROLE_RANK: Record<string, number> = {
  admin: 100,
  company_admin: 90,
  kb_admin: 70,
  dept_manager: 60,
  manager: 60,
  employee: 30,
  editor: 30,
  user: 30,
  viewer: 10,
};

/**
 * 角色中文名（与后端 permissions.ROLE_LABELS 同一份口径）。
 * 后端会下发 role_label，这里只是首屏兜底与下拉框展示。
 */
export const ROLE_LABELS: Record<string, string> = {
  admin: "平台管理员",
  company_admin: "企业管理员",
  kb_admin: "知识库管理员",
  dept_manager: "部门负责人",
  manager: "部门负责人",
  employee: "普通员工",
  editor: "普通员工",
  user: "普通员工",
  viewer: "只读成员",
};

export function roleRank(role?: string | null): number {
  return ROLE_RANK[(role ?? "").trim()] ?? ROLE_RANK.employee;
}
