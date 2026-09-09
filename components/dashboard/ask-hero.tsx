"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { FolderOpen, Globe2, Search, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useApp } from "@/lib/context/app-context";
import { suggestedQuestions, type Collection } from "@/lib/types";

interface AskHeroProps {
  collections: Collection[];
}

/**
 * 仪表盘顶部的快捷提问入口（主界面互动优化）：
 * 用户不必先跳转到对话页，在主界面直接输入问题即可开始检索问答。
 * 回车 / 点击「提问」→ /chat?q=… 由聊天页自动发送。
 */
export function AskHero({ collections }: AskHeroProps) {
  const router = useRouter();
  const { activeCollectionId } = useApp();
  const [question, setQuestion] = useState("");

  const activeCollection = collections.find(
    (c) => c.id === activeCollectionId
  );

  const ask = (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    router.push(`/chat?q=${encodeURIComponent(trimmed)}`);
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      aria-label="快捷提问"
    >
      <Card className="relative overflow-hidden bg-gradient-to-b from-primary/[0.05] to-transparent">
        <CardContent className="flex flex-col items-center gap-5 px-6 py-8 sm:px-10 sm:py-10">
          <div className="flex flex-col items-center gap-1.5 text-center">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-xs font-medium text-primary">
              <Sparkles className="size-3" />
              AI 检索问答
            </span>
            <h2 className="text-xl font-semibold tracking-tight sm:text-2xl">
              想从知识库里了解什么？
            </h2>
            <p className="text-sm text-muted-foreground">
              输入问题，AI 将从已上传的文档中检索并生成带引用来源的回答
            </p>
          </div>

          <form
            className="relative mx-auto w-full max-w-2xl"
            onSubmit={(e) => {
              e.preventDefault();
              ask(question);
            }}
          >
            <Search className="pointer-events-none absolute left-4 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="例如：总结一下产品需求文档的要点…"
              className="h-12 rounded-xl pl-11 pr-24 text-[15px]"
              aria-label="向知识库提问"
            />
            <Button
              type="submit"
              className="absolute right-1.5 top-1/2 h-9 -translate-y-1/2 rounded-lg px-4"
              disabled={!question.trim()}
            >
              提问
            </Button>
          </form>

          <div className="flex flex-wrap items-center justify-center gap-2">
            {activeCollection ? (
              <Badge variant="secondary" className="gap-1.5 py-1">
                <FolderOpen className="size-3" />
                检索范围：{activeCollection.name}
              </Badge>
            ) : (
              <Badge variant="outline" className="gap-1.5 py-1">
                <Globe2 className="size-3" />
                检索范围：全部文档
              </Badge>
            )}
            {suggestedQuestions.slice(0, 3).map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => ask(q)}
                className="rounded-full border border-border bg-background px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-foreground"
              >
                {q}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.section>
  );
}
