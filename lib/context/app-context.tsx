"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useAuth } from "@/lib/context/auth-context";

const ACTIVE_COLLECTION_KEY = "rag-active-collection";
const ACTIVE_CONVERSATION_KEY = "rag-active-conversation";
/**
 * 上面两个指针**属于哪个账号**.
 *
 * 这两个键是 localStorage 级别、跨标签页与跨登录保留的。换账号登录时会继承
 * 上一个账号的 conversation_id —— 后端按"不是我的会话"返回 404，用户看到的是
 * "历史对话打不开，还一直报 Conversation not found"。存一份归属人，账号一变
 * 就把指针丢掉，从源头断掉这条脏数据链路。
 */
const SCOPE_OWNER_KEY = "rag-active-owner";

interface AppContextValue {
  activeCollectionId: string | null;
  setActiveCollectionId: (id: string | null) => void;
  activeConversationId: string | null;
  setActiveConversationId: (id: string | null) => void;
  conversationVersion: number;
  notifyConversationChanged: () => void;
  refreshKey: number;
  refresh: () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [activeCollectionId, setActiveCollectionIdState] = useState<
    string | null
  >(null);
  const [activeConversationId, setActiveConversationIdState] = useState<
    string | null
  >(null);
  const [conversationVersion, setConversationVersion] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setActiveCollectionIdState(localStorage.getItem(ACTIVE_COLLECTION_KEY));
    setActiveConversationIdState(
      localStorage.getItem(ACTIVE_CONVERSATION_KEY)
    );
    setHydrated(true);
  }, []);

  /**
   * 账号切换即丢弃上一位用户的活动指针.
   *
   * 不销毁的话，新登录的账号会拿着**别人的** conversation_id 去拉历史：
   * 后端把不属于自己的会话一律当"不存在"返回 404，界面表现为每次进入对话页
   * 都弹一次 "加载历史对话失败：Conversation not found"，历史永远是空的。
   *
   * 只认 id 相等（不是 username）：换账号 → 清；同一账号登出再登入 → 保留，
   * 因为他确实还是那个会话的主人。老数据没有归属人记录（owner === null）时
   * 只补记录、不做清理，避免升级后把用户正在进行的会话误清。
   */
  useEffect(() => {
    if (!hydrated || !user?.id) return;
    const owner = localStorage.getItem(SCOPE_OWNER_KEY);
    if (owner === null) {
      localStorage.setItem(SCOPE_OWNER_KEY, user.id);
      return;
    }
    if (owner === user.id) return;
    localStorage.removeItem(ACTIVE_CONVERSATION_KEY);
    localStorage.removeItem(ACTIVE_COLLECTION_KEY);
    localStorage.setItem(SCOPE_OWNER_KEY, user.id);
    setActiveConversationIdState(null);
    setActiveCollectionIdState(null);
  }, [hydrated, user?.id]);

  const setActiveCollectionId = useCallback((id: string | null) => {
    setActiveCollectionIdState(id);
    if (id) localStorage.setItem(ACTIVE_COLLECTION_KEY, id);
    else localStorage.removeItem(ACTIVE_COLLECTION_KEY);
  }, []);

  const setActiveConversationId = useCallback((id: string | null) => {
    setActiveConversationIdState(id);
    if (id) localStorage.setItem(ACTIVE_CONVERSATION_KEY, id);
    else localStorage.removeItem(ACTIVE_CONVERSATION_KEY);
  }, []);

  const notifyConversationChanged = useCallback(() => {
    setConversationVersion((v) => v + 1);
  }, []);

  const refresh = useCallback(() => {
    setRefreshKey((key) => key + 1);
  }, []);

  const value = useMemo(
    () => ({
      activeCollectionId: hydrated ? activeCollectionId : null,
      setActiveCollectionId,
      activeConversationId: hydrated ? activeConversationId : null,
      setActiveConversationId,
      conversationVersion,
      notifyConversationChanged,
      refreshKey,
      refresh,
    }),
    [
      activeCollectionId,
      activeConversationId,
      conversationVersion,
      hydrated,
      notifyConversationChanged,
      refresh,
      refreshKey,
      setActiveCollectionId,
      setActiveConversationId,
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used within AppProvider");
  }
  return context;
}
