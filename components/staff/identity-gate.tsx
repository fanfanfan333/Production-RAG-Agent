"use client";

import { useEffect, useRef, useState } from "react";
import { useAuth } from "@/lib/context/auth-context";
import { IdentityDialog } from "@/components/staff/identity-dialog";

/**
 * 未验证拦截器.
 *
 * 挂在根布局里（而不是每个页面），因为 App Router 的根布局在客户端路由切换时
 * **不会重挂载** —— "本次会话已经弹过一次" 这个状态因此能保持住，不会每翻一页
 * 就再弹一次。
 *
 * 触发条件：身份状态为 none（从未验证）或 rejected（被拒后可改）
 *           —— pending（审核中）与 approved（已通过）不再打扰。
 *
 * 弹窗本身仍可被"取消"关掉；关掉后不阻断浏览，但所有业务接口都会被后端的
 * 身份闸门拦下（403），页面上会有明确提示，用户能自己回到个人主页重新发起。
 */
export function IdentityGate() {
  const { user, refreshUser } = useAuth();
  const [open, setOpen] = useState(false);
  const autoShownFor = useRef<string | null>(null);

  const status = user?.identity_status;
  const needsVerification = status === "none" || status === "rejected";

  useEffect(() => {
    if (!user || !needsVerification) return;
    // 同一账号在本次会话里只自动弹一次
    if (autoShownFor.current === user.id) return;
    autoShownFor.current = user.id;
    setOpen(true);
  }, [user, needsVerification]);

  // 登出后再登录另一个账号时，允许重新自动弹窗
  useEffect(() => {
    if (!user) autoShownFor.current = null;
  }, [user]);

  if (!user) return null;

  return (
    <IdentityDialog
      open={open}
      onOpenChange={setOpen}
      forced={needsVerification}
      onChanged={() => {
        void refreshUser();
      }}
    />
  );
}
