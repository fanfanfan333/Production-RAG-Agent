import { apiFetch } from "@/lib/api/client";
import type {
  ConversationMessage,
  ConversationSummary,
} from "@/lib/types";

function normalizeConversation(raw: Record<string, unknown>): ConversationSummary {
  return {
    id: String(raw.id ?? ""),
    title: String(raw.title ?? "新对话"),
    messageCount: Number(raw.message_count ?? raw.messageCount ?? 0),
    createdAt: raw.created_at ? String(raw.created_at) : undefined,
    updatedAt: raw.updated_at ? String(raw.updated_at) : undefined,
  };
}

export async function listConversations(limit = 50): Promise<ConversationSummary[]> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/conversations?limit=${limit}`
  );
  const list = (raw.conversations ?? []) as Record<string, unknown>[];
  return list.map(normalizeConversation);
}

export async function getConversationMessages(
  conversationId: string
): Promise<ConversationMessage[]> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/conversations/${conversationId}/messages`
  );
  const list = (raw.messages ?? []) as Record<string, unknown>[];
  return list.map((item) => ({
    role: item.role === "assistant" ? "assistant" : "user",
    content: String(item.content ?? ""),
    createdAt: item.created_at ? String(item.created_at) : undefined,
  }));
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await apiFetch(`/conversations/${conversationId}`, { method: "DELETE" });
}
