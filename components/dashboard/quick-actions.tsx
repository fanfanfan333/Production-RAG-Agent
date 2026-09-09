"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Upload,
  FileStack,
  FolderOpen,
  Settings2,
  ArrowUpRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { quickActions } from "@/lib/types";

const iconMap = {
  upload: Upload,
  documents: FileStack,
  collections: FolderOpen,
  settings: Settings2,
};

export function QuickActions() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.15 }}
    >
      <Card>
        <CardHeader>
          <CardTitle>快捷操作</CardTitle>
          <CardDescription>快速开始常用任务</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2">
          {quickActions.map((action, index) => {
            const Icon = iconMap[action.icon];

            const content = (
              <>
                <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                  <Icon className="size-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{action.label}</p>
                  <p className="mt-0.5 truncate text-xs text-muted-foreground">
                    {action.description}
                  </p>
                </div>
                <ArrowUpRight className="size-4 shrink-0 text-muted-foreground opacity-0 transition-all group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:opacity-100" />
              </>
            );

            return (
              <motion.div
                key={action.id}
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: 0.2 + index * 0.05 }}
              >
                {action.id === "upload" ? (
                  <Button
                    variant="outline"
                    className="group h-auto w-full justify-start gap-3 border-primary/30 px-4 py-4 text-left hover:bg-accent"
                    onClick={() => {
                      // 先滚动到上传卡片，再直接唤起文件选择框，
                      // 一步到位开始上传（此前只滚动不触发，体验断裂）
                      const el = document.getElementById("upload");
                      el?.scrollIntoView({ behavior: "smooth" });
                      window.setTimeout(() => {
                        document
                          .getElementById("upload-file-input")
                          ?.click();
                      }, 350);
                    }}
                  >
                    {content}
                  </Button>
                ) : (
                  <Button
                    variant="outline"
                    className="group h-auto w-full justify-start gap-3 px-4 py-4 text-left hover:bg-accent"
                    asChild
                  >
                    <Link href={action.href}>{content}</Link>
                  </Button>
                )}
              </motion.div>
            );
          })}
        </CardContent>
      </Card>
    </motion.div>
  );
}
