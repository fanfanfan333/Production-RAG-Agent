export { API_BASE, ApiError, apiFetch } from "@/lib/api/client";
export { getDocuments, deleteDocument, assignDocumentToCollection } from "@/lib/api/documents";
export { uploadDocuments } from "@/lib/api/upload";
export { streamQuery } from "@/lib/api/query";
export { getCollections, createCollection, deleteCollection } from "@/lib/api/collections";
export { getHealth } from "@/lib/api/health";
export {
  listBadCases,
  updateBadCase,
  getBadCaseStats,
  type BadCaseItem,
  type BadCaseStatus,
  type BadCaseStatsResponse,
} from "@/lib/api/badcases";
