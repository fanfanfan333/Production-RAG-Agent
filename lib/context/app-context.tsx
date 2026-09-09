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

const ACTIVE_COLLECTION_KEY = "rag-active-collection";
const ACTIVE_CONVERSATION_KEY = "rag-active-conversation";

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
