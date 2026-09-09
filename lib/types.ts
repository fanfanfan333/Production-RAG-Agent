export type DocumentStatus = "indexed" | "processing" | "failed" | "already_exists";

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

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  sources?: QuerySource[];
  isStreaming?: boolean;
  error?: string;
  /** Pipeline that produced this message (doc-relation analysis vs RAG). */
  mode?: "rag" | "doc_relations" | "list_documents";
  /** Local-only feedback state (👍/👎). */
  feedback?: FeedbackRating;
  /** Conversation id at the time this answer finished streaming. */
  conversationId?: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  messageCount: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
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
  "文档涵盖了哪些主要主题？",
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
