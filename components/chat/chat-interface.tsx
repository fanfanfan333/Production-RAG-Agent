"use client";

import {
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type TouchEvent as ReactTouchEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  AlertCircle,
  ArrowDown,
  Bot,
  Copy,
  Check,
  Download,
  FileDown,
  ImageIcon,
  Loader2,
  MessageSquarePlus,
  RotateCcw,
  Send,
  ShieldAlert,
  ShieldCheck,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { toast } from "sonner";
import { ApiError } from "@/lib/api/client";
import { streamQuery, type OutputGuardInfo } from "@/lib/api/query";
import { downloadGeneratedDocument } from "@/lib/api/documents";
import { getConversationMessages } from "@/lib/api/conversations";
import { useApp } from "@/lib/context/app-context";
import {
  isDocRelationQuery,
  isDocumentAgentQuery,
  isDocumentListQuery,
  suggestedQuestions,
  type ChatMessage,
  type CitationCheckInfo,
  type EvidenceInfo,
  type GeneratedDocumentInfo,
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

// 距底部多少像素以内算"贴底"。放在模块作用域：它们是纯常量，放进组件体内会让
// useCallback([]) 的依赖检查报警。
//
// 两个阈值刻意**不对称**（迟滞）：离开底部用大阈值、滚回底部用小阈值。用同一个
// 值会在临界点反复横跳 —— 用户向上滚一点点 → 判定已离开 → 内容又长了一点把他
// 顶回阈值内 → 判定又贴底 → 又被拽下去，表现就是"滚不动"。
// 旧实现的 80px 单阈值则太大：用户明明已经上滑了七八十像素，仍被判定为贴底
// 而当场被拽回去。
const LEAVE_BOTTOM_PX = 24; // 距底部超过这么多 → 用户已离开底部，停止跟随
const BACK_TO_BOTTOM_PX = 12; // 回到距底部这么多以内 → 用户滚回了底部，恢复跟随

// 问题1修复: 把后端真实的 intent（或前端本地预设的 mode）映射成可读徽章。
// 顺序：服务端 intent > 前端预设 mode > 兜底 RAG。
function pipelineLabel(pipeline: ChatMessage["intent"] | ChatMessage["mode"] | null): string {
  switch (pipeline) {
    case "general_chat":
      return "通用闲聊（不检索知识库）";
    case "document_summary":
      return "文档总结";
    case "doc_relations":
      return "跨文档关联分析";
    case "list_documents":
      return "知识库文档清单";
    case "document_agent":
      return "Document Agent（生成 Word 文档）";
    case "knowledge_qa":
    case "rag":
    case null:
    case undefined:
      return "混合检索 + 精排重排序";
    default:
      return "混合检索 + 精排重排序";
  }
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

// LaTeX 公式归一化与流式半截公式隐藏：实现在 lib/math-normalize.ts（与测试页共用）。
import { normalizeMath, hidePartialMath } from "@/lib/math-normalize";

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
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
    >
      {renderAnswerContent(content, hasSources, isStreaming)}
    </ReactMarkdown>
  );
});

function renderAnswerContent(
  content: string,
  hasSources: boolean,
  isStreaming: boolean
): string {
  let text = content;

  // 公式处理：先藏住流式中的半截公式，再归一化各种 LaTeX 定界符。
  if (isStreaming) text = hidePartialMath(text);
  text = normalizeMath(text);

  if (!hasSources) return text;

  // While streaming, hide a partially-arrived marker at the end
  // (e.g. "[Sou" / "[Source 1") to avoid flickering raw brackets.
  if (isStreaming) {
    text = text.replace(/\[S?o?u?r?c?e?\s*\d*$/, "").replace(/\[\d*$/, "");
  }

  // 引用标记 → **可点击上标**：`[Source 1]` 渲染成 `[¹](#source-1)`，
  // 点击即跳到下方第 1 张引用卡片（卡片上有 id="source-1" 并高亮）。
  //
  // 之前只做上标替换、不生成链接，结果是"细粒度引用"只能看不能点：
  // 用户读到一句带 ³ 的结论，想核对该条证据时必须自己在下面数到第 3 张卡片。
  // 行号已经精确到"第几行"，却差最后一步跳转，等于白给。
  text = text.replace(
    /\[Source\s*(\d+)(?:\s*[-–~]\s*(\d+))?\]/gi,
    (_m, a: string, b?: string) => {
      const num = Number(a);
      const label =
        toSuperscript(num) +
        (b ? `\u207B${toSuperscript(Number(b))}` : "");
      return `[${label}](#source-${num})`;
    }
  );
  // 裸 [N]（1–30）同样视作引用编号，但排除两种 Markdown 结构：
  //   `[1](url)`  —— 链接语法，紧跟 "("
  //   `[1]: url`  —— 引用式链接定义，紧跟 ":"
  // 否则会把模型输出的定义行改成链接、破坏文档结构。
  text = text.replace(/\[(\d{1,2})\](?![:(])/g, (m, n: string) => {
    const num = Number(n);
    if (num < 1 || num > 30) return m;
    return `[${toSuperscript(num)}](#source-${num})`;
  });
  return text;
}

/** Document Agent 产物卡片：一键下载生成的 Word 文档（最终效果）。 */
function DocumentCard({ doc }: { doc: GeneratedDocumentInfo }) {
  const [busy, setBusy] = useState(false);
  const ok = !doc.error && !!doc.filename;
  const sizeKb = doc.size_bytes
    ? Math.max(1, Math.round(doc.size_bytes / 1024))
    : 0;

  return (
    <div className="mt-3 rounded-xl border border-primary/25 bg-primary/5 p-3">
      <div className="flex items-start gap-2">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <FileDown className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium text-foreground/90">
            {doc.title || "生成的文档"}
          </p>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            {ok
              ? `Word 文档 · ${doc.section_count} 小节 · ${doc.table_count} 表格 · ${doc.image_count} 图 · ${sizeKb} KB`
              : `生成失败：${doc.error ?? "未知原因"}`}
          </p>
        </div>
        {ok && (
          <Button
            size="sm"
            variant="outline"
            className="h-7 shrink-0 px-2 text-xs"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await downloadGeneratedDocument(doc.download_url, doc.filename);
                toast.success("文档已开始下载");
              } catch (err) {
                toast.error(err instanceof Error ? err.message : "下载失败");
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Download className="size-3.5" />
            )}
            下载 .docx
          </Button>
        )}
      </div>
    </div>
  );
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
  // 真正的滚动容器。自动吸底必须操作**它**，不能用 scrollIntoView ——
  // scrollIntoView 会连带滚动所有祖先容器，在嵌套布局里会把整页顶走。
  const scrollRef = useRef<HTMLDivElement>(null);
  // "是否贴底"。用 ref 而不是只靠 state：吸底判断发生在 effect 里，需要读到
  // 最新值却不想因此重新订阅 effect（否则每次滚动都会重跑吸底逻辑）。
  const pinnedRef = useRef(true);
  // 上一次"自动吸底"落到的 scrollTop。**这是判断"用户有没有自己滚过"最可靠的
  // 信号**：内容变长只改 scrollHeight、不改 scrollTop，所以只要这个值没变，就
  // 一定不是用户滚的 —— 哪怕某一次渲染让内容一次性长高了 300px（一次性渲染出
  // 引用来源列表时就会），也不该把忠实贴底的用户误判成"已离开底部"。
  const autoScrollTopRef = useRef<number | null>(null);
  // "回到最新"的平滑滚动进行中。这段时间里 scroll 事件会持续上报"离底部还很远"，
  // 不能让它把刚点亮的 pinnedRef 又刷成 false（否则点完按钮反被判定成"用户滚
  // 上去了"）。
  const smoothScrollingRef = useRef(false);
  // 触摸起点 Y：手指下滑（clientY 变大）= 想看上面的历史。
  const touchStartYRef = useRef<number | null>(null);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  // 上一次的消息条数：用来区分"新增了一条消息"和"同一条消息在流式增长"。
  const prevCountRef = useRef(0);
  const streamingRef = useRef(false);
  // Which conversation the current `messages` state belongs to. Prevents the
  // restore effect from clobbering freshly streamed messages (sources /
  // thinking live only in memory) when the done event updates the context.
  const loadedConvRef = useRef<string | null>(null);
  // 还原请求的代次号：只有"最后一次"请求的结果允许写入视图，避免快速切换
  // 会话时旧响应后到、把新会话的消息覆盖掉。
  const restoreSeqRef = useRef(0);
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
      lines.push(`## ${msg.role === "user" ? "提问" : "回答"}`);
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

  // 用户主动往上滚（滚轮 / 触摸 / 方向键）→ **同步**解除吸底。
  //
  // 只靠 onScroll 不够 —— 这正是本 BUG 的根因：滚轮与触控板的滚动是浏览器在随后
  // 几帧里陆续应用的，scroll 事件因此晚于用户输入；而流式回答时 token 每几十毫秒
  // 就来一次，吸底 effect 跑得比 scroll 事件还勤。于是 effect 常常赶在 scroll 事件
  // 送达之前，拿着"还没来得及变 false 的 pinnedRef"把 scrollTop 拉回底部 —— 用户
  // 那一次向上滚当场被抹掉，pinnedRef 也就永远等不到变 false 的机会，表现就是
  // "AI 回答时鼠标完全滚不动"。wheel / touchmove / keydown 都是同步派发的输入事件，
  // 在这里立刻解除吸底，才能在下一个 token 到达前把状态定下来。
  //
  // ⚠️ 这里**不能**用"当前是否贴底"当守卫。曾经写成
  // `if (scrollHeight - scrollTop - clientHeight <= BACK_TO_BOTTOM_PX) return;`，
  // 看着像"已经贴底就别解除，省得『回到最新』按钮在底部反复闪"，其实与自身存在
  // 的理由直接矛盾：wheel 是**同步**派发的，事件到达的那一刻这一帧的滚动还没落地，
  // DOM 依然读得出"贴底"。于是用户从底部开始上滑的第一下必然命中这个 return ——
  // 守卫恰好把自己唯一该留下来的那次给挡掉了，同步解除从未发生，BUG 原样复现。
  // 判据必须是**输入**（用户做了什么），不能是**结果位置**（DOM 现在读起来怎样）。
  // 抖动过滤因此上移到调用方，按输入幅度阈值做，而不是在这里按位置做。
  const releasePin = useCallback(() => {
    // 用户一旦有输入，就不再处于"回到最新"的平滑滚动中
    smoothScrollingRef.current = false;
    pinnedRef.current = false;
    setShowJumpToLatest(true);
  }, []);

  const handleWheel = useCallback(
    (event: ReactWheelEvent<HTMLDivElement>) => {
      // 抖动过滤在这里做（按**输入幅度**，理由见 releasePin 的注释）：
      // deltaY < 0 = 向上滚（看历史）；触控板惯性回弹会吐出 -1~-7 的噪声，
      // 不值得为它解除吸底。
      if (event.deltaY <= -8) releasePin();
    },
    [releasePin]
  );

  const handleTouchStart = useCallback(
    (event: ReactTouchEvent<HTMLDivElement>) => {
      const touch = event.touches[0];
      if (!touch) return;
      touchStartYRef.current = touch.clientY;
    },
    []
  );

  const handleTouchMove = useCallback(
    (event: ReactTouchEvent<HTMLDivElement>) => {
      const start = touchStartYRef.current;
      const touch = event.touches[0];
      if (start === null || !touch) return;
      // 手指下滑（clientY 变大）= 想看上面的历史。同样按输入幅度过滤：
      // 位移 > 8px 才解除，抹掉手指微抖。
      if (touch.clientY - start > 8) releasePin();
    },
    [releasePin]
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      // 容器里有可聚焦元素（引用锚点、按钮），方向键 / PageUp / Home 会滚动它
      if (
        event.key === "ArrowUp" ||
        event.key === "PageUp" ||
        event.key === "Home"
      ) {
        // 方向键 / PageUp / Home 本身就是明确的离散输入，不需要抖动阈值，
        // 一次按键 = 一次明确意图，无条件解除。
        releasePin();
      }
    },
    [releasePin]
  );

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;

    // 我们自己发起的平滑滚动（点"回到最新"）途中不做"离开底部"判定，
    // 否则动画期间持续上报的大距离会把刚点亮的 pinnedRef 又刷成 false。
    if (smoothScrollingRef.current) {
      if (distance <= BACK_TO_BOTTOM_PX) smoothScrollingRef.current = false;
      autoScrollTopRef.current = el.scrollTop;
      return;
    }

    // 迟滞：离开用大阈值、回来用小阈值，中间地带保持原状态不横跳。
    if (distance > LEAVE_BOTTOM_PX) {
      pinnedRef.current = false;
      setShowJumpToLatest(true);
    } else if (distance <= BACK_TO_BOTTOM_PX) {
      pinnedRef.current = true;
      // 用户自己滚回底部时，立刻收起"回到最新"按钮
      setShowJumpToLatest(false);
    }
    // 记住当前位置：下一次吸底 effect 用它判断"是不是用户滚的"
    autoScrollTopRef.current = el.scrollTop;
  }, []);

  const jumpToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = true;
    smoothScrollingRef.current = true;
    setShowJumpToLatest(false);
    // 只有**用户点击**这条路径用 smooth：流式跟随是 instant 的（见下面吸底 effect
    // 的第 3 条），token 一到，effect 的 instant 定位会当场把正在跑的 smooth 动画
    // 打断 —— 用户看到的是直接瞬移到底，而不是剩下的那半段缓动。所以 smooth 在
    // 流式期间事实上无效，别指望它。
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, []);

  // 自动吸底。
  //
  // 旧写法是 `scrollIntoView({behavior:"smooth"})`，依赖数组
  // 是 [messages] —— 而流式回答时 messages **每个 token 都会变**，于是每个 token
  // 都触发一次平滑滚动，把视口强行拽到底部。用户往上滚就被立刻拉回去，
  // 表现就是"AI 回答时鼠标滚不动"。这里做四件事修掉它：
  //
  // 1. **只在用户本就贴底时才跟随**（pinnedRef）。用户滚上去读历史 → 不打扰，
  //    改为显示"回到最新"按钮让他自己决定什么时候回去。
  // 2. **直接设 scrollTop**，不用 scrollIntoView —— 后者会连带滚动祖先容器。
  // 3. **流式增长用即时滚动**：用 behavior:"instant" 而不是 "auto"，因为 "auto"
  //    会去读 CSS scroll-behavior，一旦哪天被设成 smooth，高频 token 就会不断
  //    打断重启动画，既追不上也费性能。
  // 4. **现场比对 scrollTop，不迷信"上一次 scroll 事件"留下的 pinnedRef**。这是
  //    本 BUG 的真正根因：滚轮 / 触控板的滚动由浏览器在随后几帧里陆续应用，scroll
  //    事件晚于用户输入；而 token 每几十毫秒来一次，effect 跑得比 scroll 事件还勤，
  //    于是它常常拿着"还没来得及变 false 的 pinnedRef"把 scrollTop 拉回底部，用户
  //    那一次向上滚当场被抹掉。现在先比对 autoScrollTopRef（我们上次吸底落在哪儿）
  //    确认"不是用户滚的"，再决定跟随。
  //
  // 例外：消息**条数增加**（用户提问 / 新一轮回答开始）时无条件回到底部 ——
  // 刚发出的消息必须可见，哪怕此前在翻历史。
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const grew = messages.length > prevCountRef.current;
    prevCountRef.current = messages.length;

    // 用户自己滚过吗？内容变长不会动 scrollTop，所以它一变，就一定是人滚的。
    const userScrolled =
      autoScrollTopRef.current !== null &&
      Math.abs(el.scrollTop - autoScrollTopRef.current) > 1;

    if (grew || (pinnedRef.current && !userScrolled)) {
      // 这里的 scrollTo 是 instant：它会无条件打断上一次「回到最新」那次
      // smooth（scrollTo 到同一个位置时更是直接变成空操作，连一个 scroll 事件
      // 都不会发）。若不在这里把标记清掉，smoothScrollingRef 会永久卡在 true，
      // 之后每个 scroll 事件都走"平滑滚动中"的早退分支，pinnedRef 再也不会被
      // 刷新 —— 表现为整个会话里都被强制吸底。
      smoothScrollingRef.current = false;
      el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
      pinnedRef.current = true;
      autoScrollTopRef.current = el.scrollTop;
      setShowJumpToLatest(false);
      return;
    }

    // 内容在长、但用户已经滚上去看历史 → 不抢他的滚动位置
    pinnedRef.current = false;
    autoScrollTopRef.current = el.scrollTop;
    setShowJumpToLatest(true);
  }, [messages]);

  // Restore the conversation when the active conversation changes
  // (e.g. user navigated away and came back, or picked one from history).
  //
  // 这段有两个曾经让"回到对话就看不到历史"的坑，务必保持现在的写法：
  //
  // 1. **不能提前把 loadedConvRef 标成已加载**。早期版本一进 effect 就
  //    `loadedConvRef.current = activeConversationId`，而 React 严格模式
  //    （next dev 默认开启）会把挂载期的 effect 跑两遍：首跑发请求 → 立刻被
  //    cleanup 标记取消 → 次跑发现 ref 已相等直接 return。结果是**没有任何
  //    一次请求的结果被写入视图**，历史永远渲染不出来。现在只有请求真正
  //    成功才打标记。
  // 2. **不能把整段历史丢掉**。只映射 role/content 会让引用来源、引用校验、
  //    证据门控等"依据快照"消失 —— 它们现在随 messages.meta 一起回传。
  useEffect(() => {
    if (streamingRef.current) return; // don't clobber an in-flight stream

    if (!activeConversationId) {
      // 没有活动会话（例如点了"新对话"）→ 清空视图。仅在确实存在过会话时
      // 才清，避免把"刚发出的第一问、conversationId 尚未回来"误清掉。
      if (loadedConvRef.current !== null) {
        loadedConvRef.current = null;
        setMessages([]);
      }
      return;
    }

    if (loadedConvRef.current === activeConversationId) return; // already shown

    const seq = ++restoreSeqRef.current;
    let cancelled = false;
    (async () => {
      try {
        const history = await getConversationMessages(activeConversationId);
        // 只接受"最新一次"请求的结果；期间开了新流也不能覆盖。
        if (cancelled || seq !== restoreSeqRef.current || streamingRef.current) {
          return;
        }
        setMessages(
          history.map((m) => ({
            id: createId(),
            role: m.role,
            content: m.content,
            // 依据快照：不还原这几项，历史里的"数据来源"就是空的
            intent: m.intent,
            sources: m.sources,
            citationCheck: m.citationCheck,
            evidence: m.evidence,
            multimodal: m.multimodal,
            outputGuard: m.outputGuard,
            document: m.document,
          }))
        );
        // 只有真正加载成功才标记，使失败（或严格模式下的首跑取消）后
        // 仍然可以重试。
        loadedConvRef.current = activeConversationId;
      } catch (err) {
        if (cancelled || seq !== restoreSeqRef.current) return;
        // 会话不存在（404）**不是故障**，别弹红色报错吓用户。
        //
        // 这条 404 在本项目里有三个真实来源，且都很常见：
        //   1. localStorage 里留着**上一个账号**的活动会话 id（换账号登录后必现）；
        //   2. 该对话刚被用户删掉 / 被「清空全部对话」清掉；
        //   3. 后端重建过库（会话表被清空）。
        //
        // 旧实现把 404 当作普通的加载失败弹 "加载历史对话失败：Conversation not
        // found"，同时**没有清掉那个脏 id** —— 于是每次切回对话页都再弹一次，
        // 用户看到的是"历史对话永远打不开、而且一直报错"。
        //
        // 正确做法：脏指针就地清掉（连带 localStorage），视图留空让用户重新开始。
        if (err instanceof ApiError && err.status === 404) {
          loadedConvRef.current = null;
          setMessages([]);
          setActiveConversationId(null);
          toast.info("该对话已不存在，已为你切换到新对话");
          return;
        }
        // 其余（后端离线 / 超时 / 权限）才是真故障：保留当前视图，提示可重试。
        toast.error(
          err instanceof Error
            ? `加载历史对话失败：${err.message}`
            : "加载历史对话失败，请稍后重试"
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeConversationId, setActiveConversationId]);

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
      // 问题1修复（截图1根因）:
      //   之前在这里硬编码 mode="rag"，后端 _LEGACY_MODE_ALIASES["rag"] 会
      //   变成 forced_mode="knowledge_qa"，导致 master_graph._route_node 直接
      //   跳过 LLM 意图路由 → general_chat（闲聊）永不可达、"你是谁"也被强制
      //   走 RAG 检索并错误引用文档。
      //
      //   现在：只有明确属于关联分析/文档列表的问题才预设 mode；其余全部传
      //   null，让后端 master graph 走 route_query() 由本地 qwen3 判定意图，
      //   闲聊/总结/检索/拒答 5 条分支都能正确触发。
      const mode = isDocRelationQuery(query)
        ? "doc_relations"
        : isDocumentListQuery(query)
          ? "list_documents"
          : isDocumentAgentQuery(query)
            ? "document_agent"
            : null;
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
        // 预设 mode（仅本地展示用，最终以服务端 intent 事件为准）
        mode: mode ?? undefined,
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
          // 后端 query.py: mode=null → 走 LLM 路由；显式传值才跳过路由
          mode: mode ?? undefined,
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
          // 问题1修复: 后端 LLM 路由返回的真实 intent（覆盖本地预设 mode）。
          // 闲聊/总结/检索/拒答 都能正确反映在徽章上。
          onRoute: (info) => {
            const intent = (info.intent || "") as ChatMessage["intent"];
            if (!intent) return;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, intent } : msg
              )
            );
          },
          // 问题3+4: Output Guard 审计信号 —— 展示"已合规校验"提示；
          // changed=true 时后端同时回传净化后的全文，用它替换已流式渲染的
          // 内容，保证「前端展示 == 落库文本」（流出的不安全 token 无法撤回，
          // 只能整段替换）。
          onOutputGuard: (info: OutputGuardInfo) => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantId) return msg;
                const patched: ChatMessage = { ...msg, outputGuard: info };
                return info.changed && info.sanitized_answer
                  ? { ...patched, content: info.sanitized_answer }
                  : patched;
              })
            );
          },
          // 部分5+6: multimodal_context 节点统计 —— 图文分流与 Vision 使用情况。
          onMultimodal: (info) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, multimodal: info } : msg
              )
            );
          },
          // Evidence Gate: 证据门控判定 —— passed=false 时后端已走拒答，
          // 这里只做展示（让用户知道"不是模型答不出，而是知识库里确实没有"）。
          onEvidence: (info: EvidenceInfo) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, evidence: info } : msg
              )
            );
          },
          // 答复性质：拒答时把引用来源标注为"未采用"。
          // sources 事件先于"是否拒答"发出，缺了这条修正，界面就会出现
          // "答不出来"＋"1 个引用来源"并存的矛盾画面。
          onAnswerStatus: (info) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, answerStatus: info } : msg
              )
            );
          },
          // Citation Verifier: 五项引用校验（存在/位置/支持/数字/日期）。
          // 有问题的引用标记已由后端移除；若净化改动了正文（移除标记或追加
          // 校验脚注），后端同时回传净化后全文，这里整段替换已流式渲染的内容
          // —— 流出的 token 无法撤回，只能整体覆盖，否则用户看到的引用与
          // 旁边的"引用存疑"徽标会自相矛盾。
          onCitationCheck: (info: CitationCheckInfo) => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantId) return msg;
                const patched: ChatMessage = { ...msg, citationCheck: info };
                return info.sanitizedAnswer
                  ? { ...patched, content: info.sanitizedAnswer }
                  : patched;
              })
            );
          },
          // Document Agent: 生成的 Word 文档（渲染下载卡片）。
          onDocument: (info) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, document: info } : msg
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
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            onWheel={handleWheel}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onKeyDown={handleKeyDown}
            // tabIndex={-1}：让这个滚动容器**可聚焦**（点击容器空白处即可），但
            // 不把它塞进 Tab 键序里打扰键盘用户。缺了它，onKeyDown 只在容器内
            // 的可聚焦元素（引用锚点、按钮）拿到焦点时才收得到事件，其余区域按
            // 方向键就是"没反应"，解吸底也就跟着失效。
            tabIndex={-1}
            className="min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-6"
          >
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
                          {/* 管线徽章：展示本条回答走过的检索管线（服务端 intent 优先） */}
                          {!message.isStreaming && (
                            <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                              <span className="rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] font-medium text-muted-foreground">
                                {pipelineLabel(message.intent ?? message.mode ?? null)}
                              </span>
                              {message.sources?.length ? (
                                <span className="rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] text-muted-foreground">
                                  {message.sources.length} 个引用
                                </span>
                              ) : null}
                              {/* 问题3+4: Output Guard 合规校验状态徽章 */}
                              {message.outputGuard ? (
                                <span
                                  className="rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] text-muted-foreground"
                                  title={
                                    message.outputGuard.changed
                                      ? `已净化：移除 ${message.outputGuard.citations_removed.length} 条越界引用、${message.outputGuard.leaked_phrases} 条系统词泄露、${message.outputGuard.hallucination_phrases} 条幻觉措辞、${message.outputGuard.tool_attempt_phrases} 条工具意图`
                                      : "已通过输出安全检测"
                                  }
                                >
                                  {message.outputGuard.changed
                                    ? "已合规净化"
                                    : "已合规校验"}
                                </span>
                              ) : null}
                              {/* Citation Verifier: 五项引用校验状态徽章。
                                  只展示"已核验/存疑"；具体哪条存疑由引用卡片
                                  逐条标注（见 SourceCitations 的 VerifiedBadge）。 */}
                              {message.citationCheck &&
                              message.citationCheck.overall !== "no_citations" &&
                              message.citationCheck.overall !==
                                "refused_by_model" ? (
                                <span
                                  className={cn(
                                    "inline-flex items-center gap-0.5 rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px]",
                                    message.citationCheck.overall === "verified"
                                      ? "text-emerald-600 dark:text-emerald-300"
                                      : "text-amber-600 dark:text-amber-300"
                                  )}
                                  title={(() => {
                                    const c = message.citationCheck;
                                    const parts = [
                                      `引用校验：${c.passed}/${c.total} 条通过`,
                                    ];
                                    if (c.hallucinated.length)
                                      parts.push(`不存在的来源 ${c.hallucinated.join(",")}`);
                                    if (c.misattributed.length)
                                      parts.push(`位置存疑 ${c.misattributed.join(",")}`);
                                    if (c.unsupported.length)
                                      parts.push(`原文未支持 ${c.unsupported.join(",")}`);
                                    if (c.number_mismatch.length)
                                      parts.push(`数字不一致 ${c.number_mismatch.join(",")}`);
                                    if (c.date_mismatch.length)
                                      parts.push(`日期不一致 ${c.date_mismatch.join(",")}`);
                                    return parts.join("；");
                                  })()}
                                >
                                  {message.citationCheck.overall === "verified" ? (
                                    <ShieldCheck className="size-2.5" />
                                  ) : (
                                    <ShieldAlert className="size-2.5" />
                                  )}
                                  {message.citationCheck.overall === "verified"
                                    ? `引用已核验 ${message.citationCheck.total}`
                                    : `引用存疑 ${
                                        message.citationCheck.total -
                                        message.citationCheck.passed
                                      }`}
                                </span>
                              ) : null}
                              {/* Evidence Gate: 证据门控 —— 仅在不通过（已拒答）时显示 */}
                              {message.evidence && !message.evidence.passed ? (
                                <span
                                  className="inline-flex items-center gap-0.5 rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] text-amber-600 dark:text-amber-300"
                                  title={`证据门控未通过：${message.evidence.reason}（最高精排分 ${message.evidence.top_score}，关键词覆盖率 ${Math.round(
                                    message.evidence.coverage * 100
                                  )}%）`}
                                >
                                  <ShieldAlert className="size-2.5" />
                                  证据不足
                                </span>
                              ) : null}
                              {/* 部分5: 图文分流 + 视觉理解状态徽章 */}
                              {message.multimodal &&
                              message.multimodal.image_count > 0 ? (
                                <span
                                  className="inline-flex items-center gap-0.5 rounded-full border border-border/60 bg-background/70 px-2 py-px text-[10px] text-muted-foreground"
                                  title={
                                    message.multimodal.vision_available
                                      ? `命中 ${message.multimodal.image_count} 张图片，其中 ${message.multimodal.vision_used} 张经视觉模型理解`
                                      : `命中 ${message.multimodal.image_count} 张图片（视觉模型未启用，仅按图内文字与图注作答）`
                                  }
                                >
                                  <ImageIcon className="size-2.5" />
                                  {message.multimodal.image_count} 图
                                  {message.multimodal.vision_available
                                    ? ` · 视觉 ${message.multimodal.vision_used}`
                                    : " · 仅OCR"}
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
                                    : (() => {
                                        const pipeline =
                                          message.intent ?? message.mode ?? null;
                                        switch (pipeline) {
                                          case "general_chat":
                                            return "正在思考…";
                                          case "document_summary":
                                            return "正在生成文档总结…";
                                          case "doc_relations":
                                            return "正在分析文档关联…";
                                          case "list_documents":
                                            return "正在获取文档清单…";
                                          default:
                                            return "正在检索知识库…";
                                        }
                                      })()}
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
                        <SourceCitations
                          sources={message.sources}
                          citationCheck={message.citationCheck}
                          refused={message.answerStatus?.refused}
                          refusalNote={message.answerStatus?.note}
                        />
                      )}

                      {message.document && (
                        <DocumentCard doc={message.document} />
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
              </div>
            )}

            {/* 用户滚上去读历史时的"回到最新"入口。用 sticky 而不是 absolute：
                absolute 在滚动容器里是相对内容定位的，会随内容一起滚走；
                sticky bottom-0 才能固定在可视区底部。 */}
            {showJumpToLatest && (
              <div className="pointer-events-none sticky bottom-0 z-10 flex justify-center pb-1">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={jumpToLatest}
                  className="pointer-events-auto gap-1.5 rounded-full border border-border/60 shadow-sm"
                >
                  <ArrowDown className="size-3.5" />
                  回到最新
                </Button>
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
              检索管线：查询改写 → 混合召回（向量 + BM25 RRF）→ Cross-Encoder 精排 → 图文分流（图片经 Vision 理解）→ 引用溯源生成
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
