import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "刚刚";
  if (diffMins < 60) return `${diffMins} 分钟前`;
  if (diffHours < 24) return `${diffHours} 小时前`;
  if (diffDays < 7) return `${diffDays} 天前`;
  return date.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

/**
 * 把后端 `current_stage` 翻成一句人话，给"处理中"的文档用。
 *
 * 入库是异步的，一份大文档要几分钟。只转一个圈的话，用户唯一能做的就是反复
 * 刷新并怀疑它卡死了；说出"正在解析内容与图片"和"正在生成向量 42%"，等待
 * 就变成了可理解的过程。
 *
 * 阶段来自 `document_service._run_ingestion`：
 * pending → parsing → chunking → checking_existing_vectors
 *         → embedding_and_indexing → completed / failed
 *
 * 返回 null 表示无需展示（已结束或阶段未知）。
 */
export function describeIngestStage(
  stage?: string | null,
  progress?: number | null
): string | null {
  switch (stage) {
    case "pending":
      return "已排队，等待开始";
    case "parsing":
      return "正在解析内容与图片";
    case "chunking":
      return "正在切分文本块";
    case "checking_existing_vectors":
      return "正在核对已有索引";
    case "embedding_and_indexing":
      return progress != null && progress > 0
        ? `正在生成向量 ${progress}%`
        : "正在生成向量";
    default:
      return null;
  }
}
