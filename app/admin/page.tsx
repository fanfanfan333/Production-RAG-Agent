"use client";

import { ShieldCheck, Users } from "lucide-react";
import { useAuth } from "@/lib/context/auth-context";
import { AppShell } from "@/components/layout/app-shell";
import { MemberPanel } from "@/components/staff/member-panel";
import { StaffReviewPanel } from "@/components/staff/staff-review-panel";
import { IdentityBadge } from "@/components/staff/identity-badge";

/**
 * 企业管理后台.
 *
 * 两个区块按权限分别渲染（能力名与后端权限矩阵一致，不另造一套）：
 *   身份验证审核   需要 staff.review —— 部门负责人及以上（可越级审核下级）
 *   成员管理       需要 staff.admin  —— 知识库管理员及以上（可更换成员职责）
 *
 * 公司隔离在后端强制：非平台管理员只会拿到本公司的申请与成员，这里不做任何
 * "客户端过滤"式的假隔离 —— 界面拿不到的数据，后端也不会下发。
 */
export default function AdminPage() {
  const { user } = useAuth();

  const isPlatformAdmin =
    user?.role === "admin" || user?.permissions?.includes("*");
  const canReview =
    isPlatformAdmin || Boolean(user?.permissions?.includes("staff.review"));
  const canAdminister =
    isPlatformAdmin || Boolean(user?.permissions?.includes("staff.admin"));

  const status = user?.identity_status ?? "approved";

  return (
    <AppShell activePath="/admin">
      <div className="mb-8">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
            企业管理后台
          </h1>
          <IdentityBadge status={status} />
        </div>
        <p className="mt-1 text-muted-foreground">
          {isPlatformAdmin
            ? "管理所有公司的知识库管理员与部门负责人，并审核全部身份验证申请"
            : "审核下级成员的身份验证申请，管理本公司成员的公司、部门与职责"}
        </p>
      </div>

      {!canReview && !canAdminister ? (
        <div className="flex flex-col items-center gap-3 rounded-xl border border-border/60 bg-card px-6 py-16 text-center">
          <ShieldCheck className="size-6 text-muted-foreground" />
          <p className="text-sm font-medium">你没有企业管理后台的访问权限</p>
          <p className="max-w-md text-[13px] text-muted-foreground">
            身份验证审核由部门负责人及以上承担，成员管理由知识库管理员及以上承担。
            如需相关权限，请联系你所在公司的知识库管理员或平台管理员。
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {canReview && <StaffReviewPanel />}
          {canAdminister && <MemberPanel />}
          {!canAdminister && canReview && (
            <p className="flex items-center justify-center gap-2 text-[12px] text-muted-foreground">
              <Users className="size-3.5" />
              成员管理需要知识库管理员及以上权限
            </p>
          )}
        </div>
      )}
    </AppShell>
  );
}
