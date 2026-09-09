import { AppShell } from "@/components/layout/app-shell";
import { ChatInterface } from "@/components/chat/chat-interface";
import { ConversationSidebar } from "@/components/chat/conversation-sidebar";

export default function ChatPage() {
  return (
    <AppShell activePath="/chat">
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">对话</h1>
        <p className="mt-1 text-muted-foreground">
          基于 AI 的知识库智能问答
        </p>
      </div>

      {/* 提问区在左，历史对话在右（深色高级感面板）；
          窄屏时历史对话收纳到提问区下方 */}
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="min-w-0 flex-1">
          <ChatInterface />
        </div>
        <aside className="w-full shrink-0 lg:w-72 xl:w-80">
          <div className="lg:sticky lg:top-6 lg:h-[calc(100vh-12rem)]">
            <ConversationSidebar />
          </div>
        </aside>
      </div>
    </AppShell>
  );
}
