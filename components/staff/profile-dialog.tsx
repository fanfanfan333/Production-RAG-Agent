"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Building2,
  ChevronDown,
  IdCard,
  KeyRound,
  Loader2,
  LogOut,
  Mail,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { changePassword } from "@/lib/api/auth";
import { fetchStaffProfile } from "@/lib/api/staff";
import { useAuth } from "@/lib/context/auth-context";
import type { StaffProfile } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { IdentityBadge } from "@/components/staff/identity-badge";
import { IdentityDialog } from "@/components/staff/identity-dialog";
import { cn } from "@/lib/utils";

/**
 * 个人主页（点击头像弹出）.
 *
 * 内容口径按产品要求：姓名 / 账号 / 公司 / 部门 / 职位，并把「修改密码」
 * 从系统设置页收进这里 —— 改密码是"关于我"的操作，放在设置的服务参数里
 * 语义不搭，用户也很难找到。（设置页仍保留入口，两处调用同一个接口。）
 *
 * 「身份验证」行直接显示状态并作为入口：未验证时点开弹窗与首页强制弹窗
 * 完全一致（同一个 IdentityDialog 组件），文案不会出现两套。
 */
export function ProfileDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { logout, refreshUser } = useAuth();
  const [profile, setProfile] = useState<StaffProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [identityOpen, setIdentityOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setProfile(await fetchStaffProfile());
    } catch {
      setProfile(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  const initial = (profile?.name || profile?.username || "··").slice(0, 2).toUpperCase();

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-md gap-0 p-0">
          <DialogHeader className="sr-only">
            <DialogTitle>个人主页</DialogTitle>
            <DialogDescription>账号信息与身份验证</DialogDescription>
          </DialogHeader>

          {loading ? (
            <div className="flex items-center justify-center py-16 text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
            </div>
          ) : !profile ? (
            <div className="px-6 py-10 text-center text-[13px] text-muted-foreground">
              无法获取账号信息，请稍后重试
            </div>
          ) : (
            <>
              {/* ── 头像 + 姓名 ─────────────────────────────────────────── */}
              <div className="flex items-center gap-3.5 border-b border-border/60 px-6 py-5">
                <div className="flex size-12 shrink-0 items-center justify-center rounded-full bg-primary text-[15px] font-medium text-primary-foreground">
                  {initial}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[15px] font-semibold">{profile.name}</p>
                  <p className="truncate text-[12px] text-muted-foreground">
                    {profile.username} · {profile.roleLabel}
                  </p>
                </div>
                <IdentityBadge
                  status={profile.identityStatus}
                  onClick={() => setIdentityOpen(true)}
                />
              </div>

              {/* ── 姓名 / 账号 / 公司 / 部门 / 职位 ─────────────────────── */}
              <dl className="space-y-2.5 px-6 py-5 text-[13px]">
                <ProfileRow icon={<IdCard className="size-3.5" />} label="姓名" value={profile.name} />
                <ProfileRow icon={<Mail className="size-3.5" />} label="账号" value={profile.username} />
                <ProfileRow
                  icon={<Building2 className="size-3.5" />}
                  label="公司"
                  // 平台管理员不属于任何一家公司（后端 company_name 返回「全平台」），
                  // 这里补一句范围说明，避免被误读成"一家叫全平台的公司"。
                  value={
                    profile.isAdmin
                      ? `${profile.companyName || "全平台"}（所有公司）`
                      : profile.companyName
                  }
                  muted={!profile.isAdmin && profile.identityStatus !== "approved"}
                />
                <ProfileRow
                  icon={<Users className="size-3.5" />}
                  label="部门"
                  value={profile.departmentName}
                />
                <ProfileRow
                  icon={<KeyRound className="size-3.5" />}
                  label="职位"
                  value={
                    profile.jobTitle
                      ? `${profile.jobTitle}（${profile.roleLabel}）`
                      : profile.roleLabel
                  }
                />
              </dl>

              {/* ── 身份验证 ───────────────────────────────────────────── */}
              <button
                type="button"
                onClick={() => setIdentityOpen(true)}
                className={cn(
                  "flex w-full items-center justify-between border-t border-border/60 px-6 py-3.5",
                  "text-[13px] transition-colors hover:bg-muted/40"
                )}
              >
                <span className="flex items-center gap-2">
                  <IdCard className="size-3.5 text-muted-foreground" />
                  身份验证
                </span>
                <IdentityBadge status={profile.identityStatus} />
              </button>

              <PasswordSection />

              {/* ── 退出登录 ───────────────────────────────────────────── */}
              <div className="border-t border-border/60 px-6 py-4">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full gap-2 text-muted-foreground hover:text-destructive"
                  onClick={logout}
                >
                  <LogOut className="size-4" />
                  退出登录
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* 与首页强制弹窗是同一个组件，状态与文案完全一致 */}
      <IdentityDialog
        open={identityOpen}
        onOpenChange={setIdentityOpen}
        onChanged={() => {
          void load();
          void refreshUser();
        }}
      />
    </>
  );
}

function ProfileRow({
  icon,
  label,
  value,
  muted,
}: {
  icon: React.ReactNode;
  label: string;
  value?: string | null;
  muted?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="flex w-[64px] shrink-0 items-center gap-1.5 text-muted-foreground">
        {icon}
        {label}
      </span>
      <span
        className={cn(
          "min-w-0 flex-1 truncate font-medium",
          (!value || muted) && "font-normal text-muted-foreground"
        )}
      >
        {value || "未设置"}
      </span>
    </div>
  );
}

/** 修改密码（默认折叠，避免个人主页一打开就是三个密码框）。 */
function PasswordSection() {
  const [expanded, setExpanded] = useState(false);
  const [oldPw, setOldPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    setError(null);
    if (!oldPw || !newPw || !confirmPw) {
      setError("请填写所有密码输入框");
      return;
    }
    if (newPw.length < 8) {
      setError("新密码至少需要 8 个字符");
      return;
    }
    if (newPw !== confirmPw) {
      setError("两次输入的新密码不一致");
      return;
    }
    if (newPw === oldPw) {
      setError("新密码不能与当前密码相同");
      return;
    }

    setSubmitting(true);
    try {
      await changePassword(oldPw, newPw);
      toast.success("密码修改成功");
      setOldPw("");
      setNewPw("");
      setConfirmPw("");
      setExpanded(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "修改失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="border-t border-border/60">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between px-6 py-3.5 text-[13px] transition-colors hover:bg-muted/40"
      >
        <span className="flex items-center gap-2">
          <KeyRound className="size-3.5 text-muted-foreground" />
          修改密码
        </span>
        <ChevronDown
          className={cn(
            "size-3.5 text-muted-foreground transition-transform",
            expanded && "rotate-180"
          )}
        />
      </button>

      {expanded && (
        <div className="space-y-3 px-6 pb-5">
          <Input
            type="password"
            autoComplete="current-password"
            placeholder="当前密码"
            value={oldPw}
            onChange={(e) => setOldPw(e.target.value)}
            disabled={submitting}
          />
          <Input
            type="password"
            autoComplete="new-password"
            placeholder="新密码（至少 8 个字符）"
            value={newPw}
            onChange={(e) => setNewPw(e.target.value)}
            disabled={submitting}
          />
          <Input
            type="password"
            autoComplete="new-password"
            placeholder="确认新密码"
            value={confirmPw}
            onChange={(e) => setConfirmPw(e.target.value)}
            disabled={submitting}
          />
          {error && <p className="text-[12px] text-destructive">{error}</p>}
          <Button
            type="button"
            className="w-full gap-2"
            onClick={submit}
            disabled={submitting}
          >
            {submitting && <Loader2 className="size-4 animate-spin" />}
            确认修改
          </Button>
        </div>
      )}
    </div>
  );
}
