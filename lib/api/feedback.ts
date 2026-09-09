import { apiFetch } from "@/lib/api/client";

export interface FeedbackPayload {
  rating: "up" | "down";
  question: string;
  answer: string;
  conversationId?: string | null;
  comment?: string;
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await apiFetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      rating: payload.rating,
      question: payload.question,
      answer: payload.answer,
      conversation_id: payload.conversationId ?? undefined,
      comment: payload.comment ?? undefined,
    }),
  });
}
