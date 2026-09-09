"use client";

import { memo, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertCircle,
  Bot,
  Copy,
  Check,
  Download,
  MessageSquarePlus,
  RotateCcw,
  Send,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { toast } from "sonner";
import { streamQuery } from "@/lib/api/query";
import { getConversationMessages } from "@/lib/api/conversations";
import { useApp } from "@/lib/context/app-context";
import {
  isDocRelationQuery,
  isDocumentListQuery,
  suggestedQuestions,
  type ChatMessage,
} from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { SourceCitations } from "@/components/chat/source-citations";
import { ThinkingDots } from "@/components/chat/thinking-dots";
import { cn } from "@/lib/utils";

function createId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

// Map [Source N] / [N] citation markers in the model output to elegant
// superscript numbers that pair with the citation list below the answer.
const SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹";

function toSuperscript(num: number): string {
  return String(num)
    .split("")
    .map((d) => SUPERSCRIPT_DIGITS[Number(d)] ?? d)
    .join("");
}

// Memoized markdown body (流式体验优化): while tokens stream in, every other
// message row re-renders too — but their content strings are unchanged, so
// memo skips the expensive full markdown re-parse and the stream stays smooth.
const MarkdownBlock = memo(function MarkdownBlock({
  content,
  hasSources,
  isStreaming,
}: {
  content: string;
  hasSources: boolean;
  isStreaming: boolean;
}) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]}>
      {renderAnswerContent(content, hasSources, isStreaming)}
    </ReactMarkdown>
  );
});

function renderAnswerContent(
  content: string,
  hasSources: boolean,
  isStreaming: boolean
): string {
  if (!hasSources) return content;
  let text = content;

  // While streaming, hide a partially-arrived marker at the end
  // (e.g. "[Sou" / "[Source 1") to avoid flickering raw brackets.
  if (isStreaming) {
    text = text.replace(/\[S?o?u?r?c?e?\s*\d*$/, "").replace(/\[\d*$/, "");
  }

  text = text.replace(
    /\[Source\s*(\d+)(?:\s*[-–~]\s*(\d+))?\]/gi,
    (_m, a: string, b?: string) =>
      toSuperscript(Number(a)) +
      (b ? `\u207B${toSuperscript(Number(b))}` : "")
  );
  text = text.replace(/\[(\d{1,2})\]/g, (m, n: string) => {
    const num = Number(n);
    return num >= 1 && num <= 30 ? toSuperscript(num) : m;
  });
  return text;
}

export function ChatInterface() {
  const {
    activeCollectionId,
    activeConversationId,
    setActiveConversationId,
    notifyConversationChanged,
  } = useApp();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const streamingRef = useRef(false);
  // Which conversation the current `messages` state belongs to. Prevents the
  // restore effect from clobbering freshly streamed messages (sources /
  // thinking live only in memory) when the done event updates the context.
  const loadedConvRef = useRef<string | null>(null);
  // Guards the one-shot /chat?q= auto-ask so it only fires once per mount.
  const autoAskRef = useRef(false);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success("已复制到剪贴板");
    setTimeout(() => setCopiedId(null), 2000);
  };

  // 删除单条消息（仅本地视图 —— 从当前展示中移除，不影响服务端历史）
  const deleteMessage = (id: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== id));
  };

  // 导出当前对话为 Markdown 文件（对标企业知识库的对话留档能力）
  const exportConversation = () => {
    if (messages.length === 0) return;
    const lines: string[] = ["# 知识库对话记录", ""];
    for (const msg of messages) {
      lines.push(`## ${msg.role === "user" ? "🧑 提问" : "🤖 回答"}`);
      lines.push("");
      lines.push(msg.content);
      if (msg.sources?.length) {
        lines.push("");
        lines.push("**引用来源：**");
        msg.sources.forEach((s, i) => {
          lines.push(
            `- [${i + 1}] ${s.documentName ?? "未知文档"}${
              s.page ? `（第 ${s.page} 页）` : ""
            }${typeof s.score === "number" ? ` · 相关度 ${Math.round(s.score * 100)}%` : ""}`
          );
        });
      }
      lines.push("");
      lines.push("---");
      lines.push("");
    }
    const blob = new Blob([lines.join("\n")], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `对话记录-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("对话已导出为 Markdown");
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Restore the conversation when the active conversation changes
  // (e.g. user navigated away and came back, or picked one from history).
  useEffect(() => {
    if (streamingRef.current) return; // don't clobber an in-flight stream
    if (loadedConvRef.current === activeConversationId) return; // already shown

    loadedConvRef.current = activeConversationId;

    if (!activeConversationId) {
      setMessages([]);
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const history = await getConversationMessages(activeConversationId);
        // streamingRef guard: a dashboard quick-ask (?q=) may have started a
        // stream right after this load began — don't clobber it with history.
        if (cancelled || streamingRef.current) return;
        setMessages(
          history.map((m) => ({
            id: createId(),
            role: m.role,
            content: m.content,
          }))
        );
      } catch {
        // conversation may have been deleted or backend offline — start clean
        if (!cancelled) setMessages([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeConversationId]);

  const startNewConversation = useCallback(() => {
    abortRef.current?.abort();
    setActiveConversationId(null);
    setMessages([]);
    setLastQuery(null);
  }, [setActiveConversationId]);

  const sendQuery = useCallback(
    async (query: string, retry = false) => {
      if (!query.trim() || isStreaming) return;

      setLastQuery(query);
      // Cross-document relation questions (问题1) run on the dedicated
      // doc_relations pipeline; "知识库里有哪些文档" listing questions run
      // on the deterministic document-list pipeline; everything else uses
      // normal RAG retrieval. Order mirrors the backend _detect_mode.
      const mode = isDocRelationQuery(query)
        ? "doc_relations"
        : isDocumentListQuery(query)
          ? "list_documents"
          : "rag";
      const userMessage: ChatMessage = {
        id: createId(),
        role: "user",
        content: query.trim(),
      };

      const assistantId = createId();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
        mode,
      };

      if (retry) {
        setMessages((prev) => {
          const next = [...prev];
          if (next[next.length - 1]?.role === "assistant") next.pop();
          return [...next, assistantMessage];
        });
      } else {
        setMessages((prev) => [...prev, userMessage, assistantMessage]);
      }

      setInput("");
      setIsStreaming(true);
      streamingRef.current = true;
      abortRef.current = new AbortController();

      try {
        await streamQuery({
          query: query.trim(),
          collectionId: activeCollectionId,
          conversationId: activeConversationId,
          mode,
          signal: abortRef.current.signal,
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content + token }
                  : msg
              )
            );
          },
          onThinking: (delta) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, thinking: (msg.thinking ?? "") + delta }
                  : msg
              )
            );
          },
          onSources: (sources) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, sources } : msg
              )
            );
          },
          onDone: (conversationId) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      isStreaming: false,
                      // 兜底：流结束但正文为空时给出可读提示，
                      // 避免只显示引用来源的空气泡
                      content:
                        msg.content ||
                        "（本次没有生成回答内容，请重试或换个问法。）",
                    }
                  : msg
              )
            );
            if (conversationId) {
              // Messages currently on screen already belong to this
              // conversation — mark it loaded so the restore effect skips.
              loadedConvRef.current = conversationId;
              if (conversationId !== activeConversationId) {
                setActiveConversationId(conversationId);
              }
              notifyConversationChanged();
            }
          },
          onError: (error) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      isStreaming: false,
                      error: error.message,
                      content: msg.content || "出了点问题，请重试。",
                    }
                  : msg
              )
            );
          },
        });
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        const message =
          err instanceof Error ? err.message : "获取回复失败";
        toast.error(message);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? { ...msg, isStreaming: false, error: message }
              : msg
          )
        );
      } finally {
        setIsStreaming(false);
        streamingRef.current = false;
        abortRef.current = null;
      }
    },
    [
      activeCollectionId,
      activeConversationId,
      isStreaming,
      notifyConversationChanged,
      setActiveConversationId,
    ]
  );

  const stopGeneration = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
    streamingRef.current = false;
    setMessages((prev) =>
      prev.map((msg) =>
        msg.isStreaming ? { ...msg, isStreaming: false } : msg
      )
    );
  };

  // Dashboard quick-ask entry (主界面互动优化):
  //   /chat?q=…     — auto-send the question in a fresh conversation
  //   /chat?draft=… — prefill the input for editing (e.g. "针对此文档提问")
  // Reads window.location.search directly (no useSearchParams) so the page
  // stays prerenderable without a Suspense boundary.
  useEffect(() => {
    if (autoAskRef.current) return;
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q")?.trim();
    const draft = params.get("draft")?.trim();
    if (!q && !draft) return;
    autoAskRef.current = true;
    window.history.replaceState(null, "", "/chat");

    if (q) {
      // Start from a clean conversation so the dashboard question gets its
      // own thread instead of continuing the previously active one.
      setActiveConversationId(null);
      loadedConvRef.current = null;
      setMessages([]);
      sendQuery(q);
    } else if (draft) {
      setInput(draft);
    }
  }, [sendQuery, setActiveConversationId]);

  return (
    <div className="flex h-[calc(100vh-12rem)] flex-col gap-4">
      <Card className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between border-b border-border/60 pb-4">
          <div>
            <CardTitle>对话</CardTitle>
            <CardDescription>
              针对已上传的文档提问
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={exportConversation}
              disabled={messages.length === 0}
              title="导出当前对话为 Markdown"
            >
              <Download className="size-4" />
              导出对话
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={startNewConversation}
              disabled={isStreaming}
            >
              <MessageSquarePlus className="size-4" />
              新对话
            </Button>
          </div>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col p-0">
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-6">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
                <div className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                  <Bot className="size-7" />
                </div>
                <div>
                  <p className="font-medium">开始对话</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    向知识库提问任何问题
                  </p>
                </div>
                <div className="flex flex-wrap justify-center gap-2">
                  {suggestedQuestions.map((question) => (
                    <Button
                      key={question}
                      variant="outline"
                      size="sm"
                      className="h-auto whitespace-normal px-3 py-2 text-left"
                      disabled={isStreaming}
                      onClick={() => sendQuery(question)}
                    >
                      {question}
                    </Button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={cn(
                      "flex gap-3",
                      message.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    {message.role === "assistant" && (
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <Bot className="size-4" />
                      </div>
                    )}
                    <div
                      className={cn(
                        "group/msg relative max-w-[85%] rounded-xl px-4 py-3 text-sm",
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted"
                      )}
                    >
                      {message.role === "assistant" ? (
                        <div className="prose prose-sm dark:prose-invert max-w-none relative">
                          {/* 管线徽章：展示本条回答走过的检索管线 */}
                          {!message.isStreaming && (
                            <div className="mb-1.5 flex items-center gap-1.5">
                              <span className="rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] font-medium text-muted-foreground">
                                {message.mode === "doc_relations"
                                  ? "跨文档关联分析"
                                  : message.mode === "list_documents"
                                    ? "知识库文档清单"
                                    : "混合检索 + 精排重排序"}
                              </span>
                              {message.sources?.length ? (
                                <span className="rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] text-muted-foreground">
                                  {message.sources.length} 个引用
                                </span>
                              ) : null}
                            </div>
                          )}
                          {/* Thinking state (问题4): three dots in a regular
                              wave, with a short status line — no more spinner.
                              Shows while waiting for retrieval, during
                              relation-digest collection, and while qwen3
                              streams its reasoning tokens. */}
                          {message.isStreaming && !message.content && (
                            <div className="flex items-center gap-3 py-1">
                              <ThinkingDots />
                              <span className="text-xs text-muted-foreground">
                                {message.thinking
                                  ? "正在思考…"
                                  : message.sources?.length
                                    ? "正在思考…"
                                    : message.mode === "doc_relations"
                                      ? "正在分析文档关联…"
                                      : message.mode === "list_documents"
                                        ? "正在获取文档清单…"
                                        : "正在检索知识库…"}
                              </span>
                            </div>
                          )}
                          {message.thinking && !message.content && (
                            <div className="mt-2 line-clamp-5 whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                              {message.thinking}
                            </div>
                          )}
                          {message.content && (
                            <MarkdownBlock
                              content={message.content}
                              hasSources={!!message.sources?.length}
                              isStreaming={!!message.isStreaming}
                            />
                          )}
                          {message.thinking && message.content && (
                            <details className="mt-2 border-t border-border/60 pt-2 text-xs text-muted-foreground [&_summary]:cursor-pointer">
                              <summary>思考过程（点击展开）</summary>
                              <p className="mt-1.5 whitespace-pre-wrap leading-relaxed">
                                {message.thinking}
                              </p>
                            </details>
                          )}
                          {message.isStreaming && message.content && (
                            <span className="ml-0.5 inline-block animate-pulse">
                              ▍
                            </span>
                          )}
                          {/* 悬停操作条：复制 + 删除 */}
                          {!message.isStreaming && message.content && (
                            <div className="absolute top-0 right-0 flex gap-0.5 opacity-0 group-hover/msg:opacity-100 transition-opacity">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-6 text-muted-foreground hover:text-foreground"
                                onClick={() => copyToClipboard(message.content, message.id)}
                              >
                                {copiedId === message.id ? (
                                  <Check className="size-3 text-emerald-500" />
                                ) : (
                                  <Copy className="size-3" />
                                )}
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-6 text-muted-foreground hover:text-destructive"
                                title="删除该条消息"
                                onClick={() => deleteMessage(message.id)}
                              >
                                <Trash2 className="size-3" />
                              </Button>
                            </div>
                          )}
                        </div>
                      ) : (
                        <>
                          <span className="whitespace-pre-wrap">{message.content}</span>
                          {!message.isStreaming && (
                            <button
                              type="button"
                              aria-label="删除该条消息"
                              className="absolute -left-8 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground/70 opacity-0 transition-opacity hover:text-destructive group-hover/msg:opacity-100"
                              onClick={() => deleteMessage(message.id)}
                            >
                              <Trash2 className="size-3.5" />
                            </button>
                          )}
                        </>
                      )}

                      {message.error && (
                        <div className="mt-2 flex items-center gap-2 text-xs text-destructive">
                          <AlertCircle className="size-3.5" />
                          {message.error}
                        </div>
                      )}

                      {message.sources && message.sources.length > 0 && (
                        <SourceCitations sources={message.sources} />
                      )}

                      {message.error && lastQuery && !message.isStreaming && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="mt-2 h-7 px-2 text-xs"
                          onClick={() => sendQuery(lastQuery, true)}
                        >
                          <RotateCcw className="size-3" />
                          重试
                        </Button>
                      )}
                    </div>
                    {message.role === "user" && (
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted">
                        <User className="size-4 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>

          <div className="border-t border-border/60 p-4">
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                sendQuery(input);
              }}
            >
              <div className="relative flex-1">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="输入关于文档的问题…（Enter 发送，Shift+Enter 换行）"
                  className="min-h-[44px] resize-none pr-16"
                  rows={1}
                  disabled={isStreaming}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendQuery(input);
                    }
                  }}
                />
                {input.length > 0 && (
                  <span className="pointer-events-none absolute bottom-2.5 right-3 text-[10px] tabular-nums text-muted-foreground/50">
                    {input.length}
                  </span>
                )}
              </div>
              {isStreaming ? (
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={stopGeneration}
                >
                  <Square className="size-4" />
                </Button>
              ) : (
                <Button type="submit" size="icon" disabled={!input.trim()}>
                  <Send className="size-4" />
                </Button>
              )}
            </form>
            <p className="mt-2 text-[11px] text-muted-foreground/70">
              检索管线：查询改写 → 混合召回（向量 + BM25 RRF）→ Cross-Encoder 精排 → 引用溯源生成
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
