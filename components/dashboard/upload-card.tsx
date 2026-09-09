"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { FileUp, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { uploadDocuments } from "@/lib/api/upload";
import { useApp } from "@/lib/context/app-context";
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

/** 把后端英文错误翻译为用户可读的中文提示 */
function friendlyUploadError(msg: string): string {
  if (/exceeds the maximum allowed size/i.test(msg)) return "文件超过大小限制（最大 50 MB）";
  if (/unsupported extension/i.test(msg)) return "不支持的文件格式";
  if (/content does not match extension/i.test(msg)) return "文件内容与扩展名不符，文件可能已损坏";
  if (/is empty/i.test(msg)) return "空文件，无法处理";
  if (/Could not read file/i.test(msg)) return "文件读取失败，请重试";
  if (/Network error/i.test(msg)) return "网络错误，请确认后端服务已启动后重试";
  if (/timed out/i.test(msg)) return "处理超时，文档较大时请稍后到文档页查看索引结果";
  return msg;
}

export function UploadCard({ disabled, onUploadComplete }: UploadCardProps) {
  const { activeCollectionId, refresh } = useApp();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processing, setProcessing] = useState(false);

  const isDisabled = disabled || uploading || processing;

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.md', '.markdown', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp', '.json', '.log'];
      const uploadableFiles = Array.from(files).filter((file) => {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        return ALLOWED_EXTENSIONS.includes(ext);
      });

      if (uploadableFiles.length === 0) {
        toast.error("仅支持上传以下格式：.pdf, .docx, .pptx, .xlsx, .csv, .txt, .md, .markdown, .png, .jpg, .jpeg, .tiff, .bmp, .webp, .json, .log");
        return;
      }

      setUploading(true);
      setUploadProgress(0);

      let slowUploadTimer: NodeJS.Timeout | null = null;
      let toastId: string | number | null = null;

      slowUploadTimer = setTimeout(() => {
        toastId = toast.loading(
          "正在生成向量嵌入… 较大的文档可能需要几分钟时间。",
          { duration: Infinity }
        );
      }, 5000);

      try {
        const results = await uploadDocuments(uploadableFiles, {
          collectionId: activeCollectionId,
          onProgress: ({ progress }) => setUploadProgress(progress),
        });

        if (slowUploadTimer) clearTimeout(slowUploadTimer);
        if (toastId) toast.dismiss(toastId);

        setUploading(false);
        setProcessing(true);

        const alreadyExistsCount = results.filter((doc) => doc.status === "already_exists").length;
        const succeededCount = results.filter((doc) => doc.status === "indexed" || doc.status === "processing").length;
        const failedDocs = results.filter((doc) => doc.status === "failed");

        if (failedDocs.length > 0) {
          const errMsg = failedDocs
            .map((doc) => `${doc.name}（${friendlyUploadError(doc.error || "处理失败")}）`)
            .join("；");
          if (succeededCount > 0) {
            // 部分成功：成功的正常提示，失败的降级为黄色警告而非红色报错
            toast.success(`成功上传 ${succeededCount} 个文件`);
            toast.warning(`${failedDocs.length} 个文件失败：${errMsg}`);
          } else {
            // 全部失败才展示红色错误
            toast.error(`上传失败：${errMsg}`);
          }
          refresh();
          onUploadComplete?.();
          return;
        }

        if (alreadyExistsCount > 0 && succeededCount === 0) {
          toast.info("文档已被索引过");
        } else if (alreadyExistsCount > 0) {
          toast.success(`成功上传 ${succeededCount} 个文件，${alreadyExistsCount} 个文件已被索引过`);
        } else {
          toast.success(
            uploadableFiles.length === 1
              ? `「${uploadableFiles[0].name}」上传成功`
              : `成功上传 ${uploadableFiles.length} 个文件`
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
        setProcessing(false);
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
            将文件拖拽到此处，或从设备中浏览选择
          </CardDescription>
        </CardHeader>
        <CardContent>
          <input
            ref={inputRef}
            id="upload-file-input"
            type="file"
            accept=".pdf,application/pdf,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.pptx,application/vnd.openxmlformats-officedocument.presentationml.presentation,.xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,.csv,text/csv,.txt,text/plain,.md,text/markdown,.markdown,text/markdown,.png,image/png,.jpg,image/jpeg,.jpeg,image/jpeg,.tiff,image/tiff,.bmp,image/bmp,.webp,image/webp,.json,application/json,.log,text/plain"
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
              {uploading || processing ? (
                <Loader2 className="size-5 animate-spin" />
              ) : (
                <FileUp className="size-5" />
              )}
            </motion.div>
            <p className="text-sm font-medium">
              {uploading
                ? `上传中… ${uploadProgress}%`
                : processing
                  ? "文档处理中…"
                  : isDragging
                    ? "松开以上传"
                    : "拖拽文件到此处"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              支持 PDF、DOCX、PPTX、XLSX、CSV、TXT、MD 等格式 — 单个文件最大 50 MB
            </p>
            {(uploading || processing) && (
              <Progress
                value={uploading ? uploadProgress : undefined}
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
