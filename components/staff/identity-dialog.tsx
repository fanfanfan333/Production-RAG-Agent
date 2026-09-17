"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  Building2,
  CheckCircle2,
  Clock,
  Loader2,
  Send,
  ShieldAlert,
} from "lucide-react";
import { toast } from "sonner";
import {
  cancelStaffRequest,
  createStaffRequest,
  fetchStaffProfile,
} from "@/lib/api/staff";
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
import { cn } from "@/lib/utils";

/**
 * 「身份验证」弹窗（三行输入：公司名称 / 公司部门 / 部门职责）.
 *
 * 两种使用场景共用同一个组件，避免出现两套措辞不一致的表单：
 *
 *   forced  —— 登录后未通过验证时强制弹出（首页拦截）。不能点 X、不能点遮罩
 *              关闭；「取消」等于退出登录。理由：需求明确"没有注册和职责的
 *              不能进入"，若允许悄悄关掉，用户就会停在一个功能全 403 的空壳里。
 *   manage  —— 从个人主页点「身份验证」主动打开。取消 = 关闭。
 *
 * 状态机（与后端 staff_service 一致）：
 *   none / rejected → 展示申请表（被拒时预填上次内容 + 显示拒绝意见）
 *   pending         → 展示「审核中」，可撤回重新填写
 *   approved        → 展示「已通过」与当前身份
 */
export function IdentityDialog({
  open,
  onOpenChange,
  forced = false,
  onChanged,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  forced?: boolean;
  onChanged?: () => void;
}) {
  const { logout } = useAuth();
  const [profile, setProfile] = useState<StaffProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [company, setCompany] = useState("");
  const [department, setDepartment] = useState("");
  const [duty, setDuty] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const me = await fetchStaffProfile();
      setProfile(me);
      const last = me.latestRequest;
      if (last && last.status !== "approved" && last.status !== "cancelled") {
        setCompany(last.companyName);
        setDepartment(last.departmentName);
        setDuty(last.duty);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法获取身份信息");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      setError(null);
      load();
    }
  }, [open, load]);

  const status = profile?.identityStatus ?? "none";
  const lastRequest = profile?.latestRequest ?? null;

  const submit = async () => {
    setError(null);
    if (!company.trim()) {
      setError("请填写公司名称");
      return;
    }
    if (!department.trim()) {
      setError("请填写公司部门");
      return;
    }
    if (!duty.trim()) {
      setError("请填写部门职责");
      return;
    }

    setSubmitting(true);
    try {
      const res = await createStaffRequest({
        companyName: company.trim(),
        departmentName: department.trim(),
        duty: duty.trim(),
      });
      toast.success(res.message || "申请已提交，等待上级审核");
      await load();
      onChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  };

  const cancelRequest = async () => {
    if (!lastRequest) return;
    setCancelling(true);
    try {
      await cancelStaffRequest(lastRequest.id);
      toast.success("已撤回，可以重新填写");
      await load();
      onChanged?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "撤回失败");
    } finally {
      setCancelling(false);
    }
  };

  const handleCancel = () => {
    if (forced) {
      // 强制场景：取消 = 退出登录，避免停在一个功能全 403 的空壳里
      logout();
      return;
    }
    onOpenChange(false);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (forced && !next) return; // 强制场景不允许外部关闭路径
        onOpenChange(next);
      }}
    >
      <DialogContent
        className="max-w-md"
        hideClose={forced}
        onInteractOutside={(e) => {
          if (forced) e.preventDefault();
        }}
        onEscapeKeyDown={(e) => {
          if (forced) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>身份验证</DialogTitle>
          <DialogDescription>
            {status === "pending"
              ? "你的身份信息正在等待上级审核"
              : status === "approved"
                ? "你已完成企业身份验证"
                : "填写你在公司的归属，提交后由上级审核开通权限"}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
          </div>
        ) : status === "pending" ? (
          <PendingState
            profile={profile}
            request={lastRequest}
            cancelling={cancelling}
            onCancel={cancelRequest}
          />
        ) : status === "approved" ? (
          <ApprovedState profile={profile} />
        ) : (
          <>
            {lastRequest?.status === "rejected" && (
              <div className="flex gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2.5 text-[12px] text-destructive">
                <ShieldAlert className="mt-0.5 size-3.5 shrink-0" />
                <div>
                  <p className="font-medium">上次申请未通过</p>
                  {lastRequest.reviewerLabel && (
                    <p className="mt-0.5">
                      审核人：{lastRequest.reviewerLabel}
                    </p>
                  )}
                  {lastRequest.reviewComment && (
                    <p className="mt-0.5">意见：{lastRequest.reviewComment}</p>
                  )}
                  <p className="mt-1">可修改下方信息后重新提交。</p>
                </div>
              </div>
            )}

            <div className="space-y-4">
              <Field
                id="company"
                label="公司名称"
                icon={<Building2 className="size-3.5" />}
                placeholder="如：某某科技有限公司"
                value={company}
                onChange={setCompany}
                disabled={submitting}
              />
              <Field
                id="department"
                label="公司部门"
                placeholder="如：研发部"
                value={department}
                onChange={setDepartment}
                disabled={submitting}
              />
              <Field
                id="duty"
                label="部门职责"
                placeholder="如：嵌入式软件工程师"
                value={duty}
                onChange={setDuty}
                disabled={submitting}
              />

              {error && (
                <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
                  <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
                  {error}
                </p>
              )}
            </div>

            {/* 左下「取消」/ 右下「申请」 */}
            <div className="mt-2 flex items-center justify-between gap-3">
              <Button
                type="button"
                variant="ghost"
                className="text-muted-foreground"
                onClick={handleCancel}
                disabled={submitting}
              >
                取消
              </Button>
              <Button
                type="button"
                className="gap-2"
                onClick={submit}
                disabled={submitting}
              >
                {submitting ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Send className="size-4" />
                )}
                申请
              </Button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Field({
  id,
  label,
  placeholder,
  value,
  onChange,
  disabled,
  icon,
}: {
  id: string;
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  icon?: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor={id}
        className="flex items-center gap-1.5 text-[13px] font-medium"
      >
        {icon}
        {label}
      </label>
      <Input
        id={id}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        maxLength={128}
      />
    </div>
  );
}

function PendingState({
  profile,
  request,
  cancelling,
  onCancel,
}: {
  profile: StaffProfile | null;
  request: StaffProfile["latestRequest"];
  cancelling: boolean;
  onCancel: () => void;
}) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3.5 py-3">
        <p className="flex items-center gap-2 text-[13px] font-medium text-amber-700 dark:text-amber-400">
          <Clock className="size-3.5" />
          审核中
        </p>
        <p className="mt-1 text-[12px] text-muted-foreground">
          已提交给上级审核，通过后即可访问知识库。请耐心等待。
        </p>
      </div>

      <dl className="space-y-1.5 rounded-lg border border-border/60 bg-muted/30 px-3.5 py-3 text-[12px]">
        <Row label="公司名称" value={request?.companyName} />
        <Row label="公司部门" value={request?.departmentName} />
        <Row label="部门职责" value={request?.duty} />
        <Row label="当前职责" value={profile?.roleLabel} />
      </dl>

      <div className="flex justify-between">
        <Button
          type="button"
          variant="ghost"
          className="text-muted-foreground"
          onClick={onCancel}
          disabled={cancelling}
        >
          {cancelling && <Loader2 className="size-4 animate-spin" />}
          撤回并重新填写
        </Button>
      </div>
    </div>
  );
}

function ApprovedState({ profile }: { profile: StaffProfile | null }) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3.5 py-3">
        <p className="flex items-center gap-2 text-[13px] font-medium text-emerald-700 dark:text-emerald-400">
          <CheckCircle2 className="size-3.5" />
          已通过
        </p>
        <p className="mt-1 text-[12px] text-muted-foreground">
          你已加入 {profile?.companyName} · {profile?.departmentName}
          ，权限为「{profile?.roleLabel}」。
        </p>
      </div>

      <dl className="space-y-1.5 rounded-lg border border-border/60 bg-muted/30 px-3.5 py-3 text-[12px]">
        <Row label="公司名称" value={profile?.companyName} />
        <Row label="公司部门" value={profile?.departmentName} />
        <Row label="部门职责" value={profile?.jobTitle} />
        <Row label="权限角色" value={profile?.roleLabel} />
      </dl>
    </div>
  );
}

function Row({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={cn("truncate font-medium", !value && "text-muted-foreground")}>
        {value || "未设置"}
      </dd>
    </div>
  );
}
