"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Building2, FileUp, Loader2, Lock, Users } from "lucide-react";
import { toast } from "sonner";
import { uploadDocuments } from "@/lib/api/upload";
import { useApp } from "@/lib/context/app-context";
import { useAuth } from "@/lib/context/auth-context";
import type { AccessLevel } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface UploadCardProps {
  disabled?: boolean;
  onUploadComplete?: () => void;
}

/**
 * 上传目标层级（三层知识库）。
 * permission 为该层所需权限名 —— 由后端权限矩阵下发（user.permissions），
 * 前端据此禁用不可选的层级，不会出现"选得动但上传被拒"。
 */
const UPLOAD_TIERS: {
  level: AccessLevel;
  label: string;
  icon: typeof Lock;
  permission: string;
  hint: string;
}[] = [
  {
    level: "private",
    label: "个人知识库",
    icon: Lock,
    permission: "",
    hint: "仅你本人可见，之后可申请共享给部门或公司",
  },
  {
    level: "department",
    label: "部门知识库",
    icon: Users,
    permission: "document.publish.department",
    hint: "同部门成员按权限可检索到",
  },
  {
    level: "tenant",
    label: "公司知识库",
    icon: Building2,
    permission: "document.publish.company",
    hint: "全公司成员按权限可检索到",
  },
];

/** 与后端 `_ALLOWED_EXTENSIONS` 保持一致（含旧版 .doc）。 */
const ALLOWED_EXTENSIONS = [
  ".pdf", ".doc", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md",
  ".markdown", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".json", ".log",
];
const ACCEPT_ATTR = ALLOWED_EXTENSIONS.join(",");
const SUPPORTED_HINT =
  "支持 PDF、DOC、DOCX、PPTX、XLSX、CSV、TXT、MD 等格式 — 单个文件最大 50 MB";

/** 把后端英文错误翻译为用户可读的中文提示 */
function friendlyUploadError(msg: string): string {
  if (/exceeds the maximum allowed size/i.test(msg)) return "文件超过大小限制（最大 50 MB）";
  if (/unsupported extension/i.test(msg)) return "不支持的文件格式";
  if (/content does not match extension|内容与扩展名不符|不是有效的/i.test(msg)) {
    return "文件内容与扩展名不符，文件可能已损坏";
  }
  if (/is empty/i.test(msg)) return "空文件，无法处理";
  if (/Could not read file/i.test(msg)) return "文件读取失败，请重试";
  if (/Network error/i.test(msg)) return "网络错误，请确认后端服务已启动后重试";
  if (/timed out/i.test(msg)) return "处理超时，文档较大时请稍后到文档页查看索引结果";
  // 后端校验消息已本地化（中文），直接透传比套一层通用文案更有用
  return msg;
}

export function UploadCard({ disabled, onUploadComplete }: UploadCardProps) {
  const { activeCollectionId, refresh } = useApp();
  const { can } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  // 默认存入个人知识库（与后端默认层级一致：上传即私有，共享须显式操作）
  const [uploadLevel, setUploadLevel] = useState<AccessLevel>("private");

  // POST /upload 现在只做「受理」——校验、判重、落一条 PENDING 行，通常几百毫秒
  // 就返回；解析 / 逐图 OCR / 向量化在服务端后台跑。所以这里不再有"处理中"的
  // 阻塞态和计时器：用户提交完就能离开页面，进度由文档列表轮询 current_stage
  // 显示（"正在解析内容与图片" / "正在生成向量 42%"）。
  const isDisabled = disabled || uploading;

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const uploadableFiles = Array.from(files).filter((file) => {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        return ALLOWED_EXTENSIONS.includes(ext);
      });
      // 被过滤掉的文件必须**明确告知**：静默丢弃会让用户以为"传成功了"，
      // 之后在文档页找不到文件才回来问。
      const skipped = Array.from(files)
        .filter((file) => {
          const ext = '.' + file.name.split('.').pop()?.toLowerCase();
          return !ALLOWED_EXTENSIONS.includes(ext);
        })
        .map((file) => file.name);

      if (skipped.length) {
        toast.warning(
          `已跳过 ${skipped.length} 个不支持的文件：${skipped.join("、")}`,
          { description: `仅支持 ${ALLOWED_EXTENSIONS.join(" / ")}` }
        );
      }

      if (uploadableFiles.length === 0) {
        return;
      }

      setUploading(true);
      setUploadProgress(0);

      // 受理很快，只有"把文件字节推上去"这一步可能慢（50 MB 大文件 + 慢链路）。
      let slowUploadTimer: NodeJS.Timeout | null = null;
      let toastId: string | number | null = null;

      slowUploadTimer = setTimeout(() => {
        toastId = toast.loading("正在上传文件…", { duration: Infinity });
      }, 5000);

      try {
        const results = await uploadDocuments(uploadableFiles, {
          collectionId: activeCollectionId,
          accessLevel: uploadLevel,
          onProgress: ({ progress }) => setUploadProgress(progress),
        });

        if (slowUploadTimer) clearTimeout(slowUploadTimer);
        if (toastId) toast.dismiss(toastId);

        const alreadyExistsCount = results.filter((doc) => doc.status === "already_exists").length;
        // 受理成功 = 后端已落 PENDING 行、正在后台入库（归一化后是 "processing"）
        const acceptedCount = results.filter((doc) => doc.status === "processing").length;
        const failedDocs = results.filter((doc) => doc.status === "failed");

        if (failedDocs.length > 0) {
          const errMsg = failedDocs
            .map((doc) => `${doc.name}（${friendlyUploadError(doc.error || "处理失败")}）`)
            .join("；");
          if (acceptedCount > 0) {
            // 部分成功：成功的正常提示，失败的降级为黄色警告而非红色报错
            toast.success(`已受理 ${acceptedCount} 个文件，正在后台建立索引`);
            toast.warning(`${failedDocs.length} 个文件失败：${errMsg}`);
          } else {
            // 全部失败才展示红色错误
            toast.error(`上传失败：${errMsg}`);
          }
          refresh();
          onUploadComplete?.();
          return;
        }

        if (alreadyExistsCount > 0 && acceptedCount === 0) {
          toast.info("文档已被索引过");
        } else if (alreadyExistsCount > 0) {
          toast.success(
            `已受理 ${acceptedCount} 个文件，${alreadyExistsCount} 个文件已被索引过`
          );
        } else {
          toast.success(
            uploadableFiles.length === 1
              ? `「${uploadableFiles[0].name}」已受理`
              : `已受理 ${uploadableFiles.length} 个文件`,
            {
              description:
                "正在后台解析内容、识别图片并生成向量，可以离开此页面；进度见「文档」页。",
            }
          );
        }

        refresh();
        onUploadComplete?.();
      } catch (err) {
        if (slowUploadTimer) clearTimeout(slowUploadTimer);
        if (toastId) toast.dismiss(toastId);
        toast.error(
          friendlyUploadError(err instanceof Error ? err.message : "上传失败")
        );
      } finally {
        setUploading(false);
        setUploadProgress(0);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [activeCollectionId, onUploadComplete, refresh]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!isDisabled) setIsDragging(true);
    },
    [isDisabled]
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (isDisabled || !e.dataTransfer.files.length) return;
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles, isDisabled]
  );

  return (
    <motion.div
      id="upload"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
    >
      <Card>
        <CardHeader>
          <CardTitle>上传文档</CardTitle>
          <CardDescription>
            将文件拖拽到此处，或从设备中浏览选择；上传前可选择文档落在哪一层知识库
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* 三层知识库：上传目标层级（无权限的层级不可选，先传个人库再申请共享） */}
          <div className="mb-4 flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs text-muted-foreground">存入</span>
            {UPLOAD_TIERS.map((item) => {
              const allowed =
                item.level === "private" || can(item.permission);
              const active = uploadLevel === item.level;
              const Icon = item.icon;
              return (
                <button
                  key={item.level}
                  type="button"
                  disabled={!allowed || isDisabled}
                  title={
                    allowed
                      ? item.hint
                      : "你的角色暂无该层级的发布权限：可先上传到个人知识库，再在文档上提交「申请共享」"
                  }
                  onClick={() => setUploadLevel(item.level)}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
                    active
                      ? "border-primary/50 bg-primary/5 text-foreground"
                      : "border-border/60 text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                    !allowed && "cursor-not-allowed opacity-50 hover:bg-transparent"
                  )}
                >
                  <Icon className="size-3.5" />
                  {item.label}
                </button>
              );
            })}
          </div>

          <input
            ref={inputRef}
            id="upload-file-input"
            type="file"
            accept={ACCEPT_ATTR}
            multiple
            className="hidden"
            disabled={isDisabled}
            onChange={(e) => {
              if (e.target.files?.length) handleFiles(e.target.files);
            }}
          />
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              "relative flex min-h-[180px] flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center transition-colors duration-200",
              isDragging
                ? "border-foreground/60 bg-accent"
                : "border-border hover:border-foreground/30 hover:bg-muted/40",
              isDisabled && "pointer-events-none opacity-60"
            )}
          >
            <motion.div
              animate={
                isDragging ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }
              }
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              className="mb-4 flex size-12 items-center justify-center rounded-xl bg-muted text-muted-foreground"
            >
              {uploading ? (
                <Loader2 className="size-5 animate-spin" />
              ) : (
                <FileUp className="size-5" />
              )}
            </motion.div>
            <p className="text-sm font-medium">
              {uploading
                ? `上传中… ${uploadProgress}%`
                : isDragging
                  ? "松开以上传"
                  : "拖拽文件到此处"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {uploading
                ? "正在把文件提交给服务端；解析与向量化在后台进行，无需等待"
                : SUPPORTED_HINT}
            </p>
            {uploading && (
              <Progress
                value={uploadProgress}
                className="mt-4 h-2 w-full max-w-xs"
              />
            )}
            <Button
              className="mt-5"
              size="sm"
              disabled={isDisabled}
              onClick={() => inputRef.current?.click()}
            >
              浏览文件
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
