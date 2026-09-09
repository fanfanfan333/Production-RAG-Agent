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
import { apiFetch } from "@/lib/api/client";

export interface AuthUser {
  id: string;
  username: string;
  role: "admin" | "user";
}

interface AuthContextValue {
  user: AuthUser | null;
  /** False until the stored token has been checked once on mount. */
  hydrated: boolean;
  login: (username: string, password: string) => Promise<AuthUser>;
  register: (username: string, password: string) => Promise<AuthUser>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface TokenResponseRaw {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

function storeSession(token: string, user: AuthUser) {
  localStorage.setItem("rag_token", token);
  localStorage.setItem("rag_user", JSON.stringify(user));
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [hydrated, setHydrated] = useState(false);

  // On mount: restore the session from localStorage and validate the token
  // against /auth/me. Invalid tokens are dropped (apiFetch redirects on 401,
  // so here we just clear local state).
  useEffect(() => {
    const token = localStorage.getItem("rag_token");
    const cachedUser = localStorage.getItem("rag_user");

    if (!token) {
      setHydrated(true);
      return;
    }

    if (cachedUser) {
      try {
        setUser(JSON.parse(cachedUser) as AuthUser);
      } catch {
        localStorage.removeItem("rag_user");
      }
    }

    apiFetch<{ id: string; username: string; role: "admin" | "user" }>(
      "/auth/me",
      { timeout: 10000 }
    )
      .then((me) => {
        setUser(me);
        localStorage.setItem("rag_user", JSON.stringify(me));
      })
      .catch(() => {
        setUser(null);
        localStorage.removeItem("rag_user");
        localStorage.removeItem("rag_token");
      })
      .finally(() => setHydrated(true));
  }, []);

  const login = useCallback(
    async (username: string, password: string) => {
      const res = await apiFetch<TokenResponseRaw>("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      storeSession(res.access_token, res.user);
      setUser(res.user);
      return res.user;
    },
    []
  );

  const register = useCallback(
    async (username: string, password: string) => {
      const res = await apiFetch<TokenResponseRaw>("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      storeSession(res.access_token, res.user);
      setUser(res.user);
      return res.user;
    },
    []
  );

  const logout = useCallback(() => {
    localStorage.removeItem("rag_token");
    localStorage.removeItem("rag_user");
    setUser(null);
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
  }, []);

  const value = useMemo(
    () => ({ user, hydrated, login, register, logout }),
    [user, hydrated, login, register, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
