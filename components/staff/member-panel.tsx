"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AlertCircle,
  FileText,
  Loader2,
  MessagesSquare,
  Pencil,
  Plus,
  Search,
  ShieldAlert,
  Trash2,
  UserCog,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { createCompany, deleteCompany, previewCompanyDeletion, renameCompany } from "@/lib/api/companies";
import {
  deleteStaffMember,
  getStaffCompanies,
  getStaffMembers,
  previewStaffMemberDeletion,
  updateStaffMember,
} from "@/lib/api/staff";
import { useAuth } from "@/lib/context/auth-context";
import type {
  CompanyDeletionImpact,
  CompanyOption,
  MemberDeletionImpact,
  StaffMember,
} from "@/lib/types";
import { ROLE_LABELS, roleRank } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectItem } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { IdentityBadge } from "@/components/staff/identity-badge";
import { cn } from "@/lib/utils";

const IDENTITY_FILTERS = [
  { key: "all", label: "全部" },
  { key: "approved", label: "已通过" },
  { key: "pending", label: "审核中" },
  { key: "none", label: "未验证" },
] as const;

/**
 * 成员管理（知识库管理员及以上）.
 *
 * 公司隔离由后端强制：非平台管理员传 ``company_id`` 也会被忽略，只返回本公司
 * 成员。前端因此把公司筛选器只对平台管理员渲染出来 —— 不显示一个点了没反应的控件。
 *
 * 「更换职责」= 已入职成员调整岗位，不必再走一遍身份验证申请。约束与审核一致：
 * 只能授予严格低于自己的角色，不能修改同级或更高等级的人，也不能把自己提权。
 */
export function MemberPanel() {
  const { user } = useAuth();
  const isPlatformAdmin = Boolean(user?.role === "admin" || user?.permissions?.includes("*"));

  const [members, setMembers] = useState<StaffMember[]>([]);
  const [companies, setCompanies] = useState<CompanyOption[]>([]);
  const [companyId, setCompanyId] = useState("");
  const [keyword, setKeyword] = useState("");
  const [filter, setFilter] = useState<(typeof IDENTITY_FILTERS)[number]["key"]>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<StaffMember | null>(null);
  const [deleting, setDeleting] = useState<StaffMember | null>(null);
  // 公司注册表操作（仅平台管理员）：创建公司 / 改名 / 删除
  const [creatingCompany, setCreatingCompany] = useState(false);
  const [renamingCompany, setRenamingCompany] = useState<CompanyOption | null>(null);
  const [deletingCompany, setDeletingCompany] = useState<CompanyOption | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [list, companyList] = await Promise.all([
        getStaffMembers({
          companyId: isPlatformAdmin && companyId ? companyId : undefined,
          keyword: keyword.trim() || undefined,
        }),
        getStaffCompanies().catch(() => [] as CompanyOption[]),
      ]);
      setMembers(list);
      setCompanies(companyList);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载成员列表失败");
    } finally {
      setLoading(false);
    }
  }, [companyId, keyword, isPlatformAdmin]);

  useEffect(() => {
    load();
  }, [load]);

  const visible = useMemo(
    () =>
      filter === "all"
        ? members
        : members.filter((m) => m.identityStatus === filter),
    [members, filter]
  );

  /** 公司管理清单的计数汇总（纯计数，不引入额外请求）。 */
  const companyStats = useMemo(() => {
    const test = companies.filter((c) => c.isTest).length;
    return { total: companies.length, test, normal: companies.length - test };
  }, [companies]);

  /** 我可授予的角色（严格低于自己等级的）——与后端 can_grant_role 同规则。 */
  const myRank = roleRank(user?.role);
  const grantable = useMemo(
    () => Object.keys(ROLE_LABELS).filter((r) => roleRank(r) < myRank && r !== "user" && r !== "editor" && r !== "manager"),
    [myRank]
  );

  /**
   * 为什么这个成员不能被删除（null = 可以删）。
   *
   * 与后端 ``_deletable_member`` 的五条判定同口径，只是提前在界面上说清楚，
   * 让用户看到"按钮为什么是灰的"，而不是点了才收到 403。
   * 注意：真正不可绕过的守门永远在后端，这里只是把原因显性化。
   */
  const deleteBlockedReason = useCallback(
    (member: StaffMember): string | null => {
      if (member.isAdmin) return "平台管理员账号不可删除";
      if (user?.id && member.id === user.id) return "不能删除自己的账号";
      if (roleRank(member.role) >= myRank)
        return `只能删除等级低于自己的成员（对方：${member.roleLabel}）`;
      return null;
    },
    [myRank, user?.id]
  );

  return (
    <Card>
      <CardHeader className="gap-4">
        <div>
          <CardTitle>成员管理</CardTitle>
          <CardDescription>
            查看本公司成员的公司、部门、职责与身份状态，可直接更换成员职责或删除成员；
            删除会带走其个人知识库与历史对话，已发布到部门/公司知识库的文档保留
          </CardDescription>
        </div>

        {/* 公司注册表（仅平台管理员）：先建公司，身份验证与文档归属才能选到它 */}
        {isPlatformAdmin && (
          <div className="rounded-lg border border-border/60 bg-muted/20 px-3.5 py-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-[13px] font-medium">公司管理</p>
                  <span className="text-[11px] text-muted-foreground">
                    共 {companyStats.total} 家 · 测试 {companyStats.test} · 普通{" "}
                    {companyStats.normal}
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  列出全部已注册公司；先创建公司，新用户提交身份验证时才能选到它；
                  改名只改展示名，
                  <strong className="font-medium">成员与文档不受影响</strong>；
                  删除公司将移除其全部成员与文档，
                  <strong className="font-medium">不可恢复</strong>
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5"
                onClick={() => setCreatingCompany(true)}
              >
                <Plus className="size-3.5" />
                创建公司
              </Button>
            </div>

            {companies.length === 0 ? (
              <p className="mt-2.5 text-[12px] text-muted-foreground">
                暂无已注册公司，请先创建公司
              </p>
            ) : (
              <ul className="mt-2 divide-y divide-border/60">
                {companies.map((c) => (
                  <li
                    key={c.companyId}
                    className="flex items-center justify-between gap-3 py-1.5"
                  >
                    <span className="flex min-w-0 items-center gap-2 text-[13px]">
                      <span className="truncate font-medium">{c.companyName}</span>
                      <Badge variant={c.isTest ? "warning" : "secondary"}>
                        {c.isTest ? "测试公司" : "普通公司"}
                      </Badge>
                      <span className="text-[11px] text-muted-foreground">
                        成员 {c.memberCount}
                      </span>
                    </span>
                    <div className="flex shrink-0 items-center gap-0.5">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-1.5 text-muted-foreground hover:text-foreground"
                        disabled={!c.canRename}
                        title={
                          c.canRename
                            ? "仅修改展示名，成员与文档不受影响"
                            : "其他管理员创建的公司不可改名"
                        }
                        onClick={() => setRenamingCompany(c)}
                      >
                        <Pencil className="size-3.5" />
                        改名
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-1.5 text-muted-foreground hover:text-destructive"
                        title="删除该公司及其全部成员与文档（不可恢复）"
                        onClick={() => setDeletingCompany(c)}
                      >
                        <Trash2 className="size-3.5" />
                        删除
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[180px] flex-1">
            <Search className="absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="按账号或姓名搜索"
              className="h-9 pl-8"
            />
          </div>

          {isPlatformAdmin && companies.length > 0 && (
            <Select
              value={companyId}
              onChange={(e) => setCompanyId(e.target.value)}
              className="w-[190px]"
            >
              <SelectItem value="">全部公司（{companies.length}）</SelectItem>
              {companies.map((c) => (
                <SelectItem key={c.companyId} value={c.companyId}>
                  {c.companyName}（{c.memberCount}）
                </SelectItem>
              ))}
            </Select>
          )}

          <div className="flex items-center gap-1.5">
            {IDENTITY_FILTERS.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setFilter(item.key)}
                className={cn(
                  "rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
                  filter === item.key
                    ? "border-primary/50 bg-primary/5 text-foreground"
                    : "border-border/60 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent className="px-0">
        {error ? (
          <div className="flex items-center gap-3 px-6 py-8 text-sm text-destructive">
            <AlertCircle className="size-4 shrink-0" />
            <span>{error}</span>
          </div>
        ) : loading ? (
          <div className="space-y-3 px-6 py-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : visible.length === 0 ? (
          <p className="flex items-center justify-center gap-2 px-6 py-8 text-sm text-muted-foreground">
            <Users className="size-4" />
            没有匹配的成员
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-[13px]">
              <thead>
                <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                  <th className="px-6 py-2.5 font-medium">姓名 / 账号</th>
                  <th className="px-3 py-2.5 font-medium">公司</th>
                  <th className="px-3 py-2.5 font-medium">部门</th>
                  <th className="px-3 py-2.5 font-medium">职位</th>
                  <th className="px-3 py-2.5 font-medium">身份验证</th>
                  <th className="px-6 py-2.5 text-right font-medium">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {visible.map((member) => (
                  <tr key={member.id} className="transition-colors hover:bg-muted/30">
                    <td className="px-6 py-3">
                      <p className="font-medium">{member.name}</p>
                      <p className="font-mono text-[11px] text-muted-foreground">
                        {member.username}
                      </p>
                    </td>
                    <td className="px-3 py-3 text-muted-foreground">
                      {member.companyName}
                    </td>
                    <td className="px-3 py-3 text-muted-foreground">
                      {member.departmentName || "—"}
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-foreground/90">
                        {member.jobTitle || "—"}
                      </span>
                      <span className="ml-1.5 text-[11px] text-muted-foreground">
                        {member.roleLabel}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <IdentityBadge status={member.identityStatus} />
                    </td>
                    <td className="px-6 py-3 text-right">
                      {member.isAdmin ? (
                        <Badge variant="secondary">最高权限</Badge>
                      ) : (
                        <div className="flex items-center justify-end gap-0.5">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="gap-1.5 text-muted-foreground hover:text-foreground"
                            disabled={roleRank(member.role) >= myRank}
                            title={
                              roleRank(member.role) >= myRank
                                ? `只能调整等级低于自己的成员（对方：${member.roleLabel}）`
                                : "调整该成员的职责 / 部门 / 启用状态"
                            }
                            onClick={() => setEditing(member)}
                          >
                            <UserCog className="size-3.5" />
                            更换职责
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-8 text-muted-foreground hover:text-destructive"
                            disabled={deleteBlockedReason(member) !== null}
                            title={
                              deleteBlockedReason(member) ??
                              `删除 ${member.name} 的账号（个人数据一并删除）`
                            }
                            onClick={() => setDeleting(member)}
                          >
                            <Trash2 className="size-3.5" />
                          </Button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>

      <EditMemberDialog
        member={editing}
        grantable={grantable}
        myRank={myRank}
        onClose={() => setEditing(null)}
        onDone={async () => {
          setEditing(null);
          await load();
        }}
      />

      <DeleteMemberDialog
        member={deleting}
        onClose={() => setDeleting(null)}
        onDone={async () => {
          setDeleting(null);
          await load();
        }}
      />

      <CompanyFormDialog
        mode="create"
        open={creatingCompany}
        onClose={() => setCreatingCompany(false)}
        onDone={async () => {
          setCreatingCompany(false);
          await load();
        }}
      />

      <CompanyFormDialog
        mode="rename"
        open={renamingCompany !== null}
        company={renamingCompany}
        onClose={() => setRenamingCompany(null)}
        onDone={async () => {
          setRenamingCompany(null);
          await load();
        }}
      />

      <DeleteCompanyDialog
        company={deletingCompany}
        onClose={() => setDeletingCompany(null)}
        onDone={async () => {
          setDeletingCompany(null);
          await load();
        }}
      />
    </Card>
  );
}

/**
 * 公司注册表的表单弹窗（创建 / 改名共用）.
 *
 * 复用现有 Dialog + Input，不引入新组件文件与依赖。错误一律展示后端文案
 * （重名时后端返回 409「公司已存在」，改名重名同样 409），前端不自己编消息。
 */
function CompanyFormDialog({
  mode,
  open,
  company,
  onClose,
  onDone,
}: {
  mode: "create" | "rename";
  open: boolean;
  company?: CompanyOption | null;
  onClose: () => void;
  onDone: () => Promise<void> | void;
}) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName(mode === "rename" ? (company?.companyName ?? "") : "");
    setError(null);
  }, [open, mode, company?.companyName]);

  if (!open) return null;

  const submit = async () => {
    const value = name.trim();
    setError(null);
    if (!value) {
      setError("请填写公司名称");
      return;
    }
    setSubmitting(true);
    try {
      if (mode === "create") {
        const created = await createCompany(value);
        toast.success(`已创建公司「${created.companyName}」`);
      } else if (company) {
        const updated = await renameCompany(company.companyId, value);
        toast.success(
          `已改名为「${updated.companyName}」，成员与文档不受影响`
        );
      }
      await onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open onOpenChange={(next) => !next && !submitting && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{mode === "create" ? "创建公司" : "公司改名"}</DialogTitle>
          <DialogDescription>
            {mode === "create"
              ? "创建后该公司可作为身份验证的可选目标，并进入你的测试公司可见范围"
              : `旧名：${company?.companyName ?? "—"}　仅修改展示名，成员与文档不受影响`}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <label
              htmlFor="company-name"
              className="text-[12px] font-medium text-muted-foreground"
            >
              公司名称
            </label>
            <Input
              id="company-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="如：测试公司1"
              maxLength={128}
              disabled={submitting}
              onKeyDown={(e) => {
                if (e.key === "Enter") submit();
              }}
            />
            <p className="text-[11px] text-muted-foreground">
              公司名唯一（忽略大小写与空格）；改名不会迁移成员、文档与向量数据
            </p>
          </div>

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              {error}
            </p>
          )}

          <Button
            className="w-full gap-2"
            onClick={submit}
            disabled={submitting}
          >
            {submitting && <Loader2 className="size-4 animate-spin" />}
            {mode === "create" ? "确定创建" : "确定改名"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** 影响清单里的一行（左标签右数字，空值置灰但不隐藏 —— 让用户看清"确实为 0"）。 */
function ImpactRow({
  icon,
  label,
  count,
  unit,
  tone,
}: {
  icon: ReactNode;
  label: string;
  count: number;
  unit: string;
  tone: "danger" | "keep";
}) {
  return (
    <li className="flex items-center gap-2">
      <span
        className={cn(
          "shrink-0",
          tone === "danger" ? "text-destructive" : "text-emerald-600 dark:text-emerald-400"
        )}
      >
        {icon}
      </span>
      <span className="flex-1 text-foreground/90">{label}</span>
      <span
        className={cn(
          "font-mono tabular-nums",
          count === 0
            ? "text-muted-foreground/60"
            : tone === "danger"
              ? "font-semibold text-destructive"
              : "text-foreground/90"
        )}
      >
        {count} {unit}
      </span>
    </li>
  );
}

/**
 * 删除公司确认弹窗（平台管理员，不可恢复）.
 *
 * 打开时先向后端要一份**影响预检**：删掉这家公司会失去什么、会留下什么，
 * 数字全部来自数据库。除了红色警示条，还要求**输入公司名**才能确认 ——
 * 这是一次"删掉整家公司（含全部员工账号与三级文档）"的破坏性操作，
 * 必须防误点。预检未完成 / 失败时确认按钮保持禁用。
 */
function DeleteCompanyDialog({
  company,
  onClose,
  onDone,
}: {
  company: CompanyOption | null;
  onClose: () => void;
  onDone: () => Promise<void> | void;
}) {
  const [impact, setImpact] = useState<CompanyDeletionImpact | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [confirmName, setConfirmName] = useState("");

  const companyId = company?.companyId ?? null;

  useEffect(() => {
    if (!companyId) return;
    let cancelled = false;
    setImpact(null);
    setError(null);
    setConfirmName("");
    setLoading(true);
    (async () => {
      try {
        const data = await previewCompanyDeletion(companyId);
        if (!cancelled) setImpact(data);
      } catch (err) {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "无法获取删除影响，请稍后重试");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [companyId]);

  if (!company) return null;

  const matched = confirmName.trim() === company.companyName.trim();

  const submit = async () => {
    if (!matched) return;
    setError(null);
    setSubmitting(true);
    try {
      const { message } = await deleteCompany(company.companyId);
      toast.success(message);
      await onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && !submitting && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trash2 className="size-4 text-destructive" />
            删除公司「{company.companyName}」？
          </DialogTitle>
          <DialogDescription>
            {company.isTest ? "测试公司" : "普通公司"} · 标识 {company.companyId}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <p className="flex items-start gap-2 rounded-lg border border-destructive/25 bg-destructive/5 px-3 py-2 text-[12px] leading-relaxed text-destructive">
            <ShieldAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>
              此操作<strong>不可恢复</strong>：该公司下的
              <strong>全部员工账号会被一并删除</strong>（连同其个人数据），员工需重新注册；
              公司下的全部文档（个人 / 部门 / 公司三级）、对话与向量索引都会消失。
            </span>
          </p>

          {loading ? (
            <p className="flex items-center gap-2 py-2 text-[12px] text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" />
              正在统计该公司的数据…
            </p>
          ) : impact ? (
            <div className="space-y-3 text-[12px]">
              <div className="rounded-lg border border-border/60 bg-muted/30 px-3 py-2.5">
                <p className="mb-1.5 font-medium text-foreground/90">将被删除</p>
                <ul className="space-y-1">
                  <ImpactRow
                    icon={<Users className="size-3.5" />}
                    label="员工账号（需重新注册）"
                    count={impact.deleted.members}
                    unit="人"
                    tone="danger"
                  />
                  <ImpactRow
                    icon={<FileText className="size-3.5" />}
                    label={`文档（个人 ${impact.deleted.documentsPrivate} · 部门 ${impact.deleted.documentsDepartment} · 公司 ${impact.deleted.documentsTenant}）`}
                    count={impact.deleted.documents}
                    unit="份"
                    tone="danger"
                  />
                  <ImpactRow
                    icon={<MessagesSquare className="size-3.5" />}
                    label="历史对话"
                    count={impact.deleted.conversations}
                    unit="个"
                    tone="danger"
                  />
                  <ImpactRow
                    icon={<FileText className="size-3.5" />}
                    label="向量索引（按入库分块预计）"
                    count={impact.deleted.vectors}
                    unit="条"
                    tone="danger"
                  />
                </ul>
              </div>

              <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-3 py-2.5">
                <p className="mb-1.5 font-medium text-foreground/90">将被保留</p>
                <ul className="space-y-1">
                  <ImpactRow
                    icon={<ShieldAlert className="size-3.5" />}
                    label="审计日志与审核留痕（身份验证 / 共享申请 / 疑难案例）"
                    count={
                      impact.kept.staffRequests +
                      impact.kept.shareRequests +
                      impact.kept.badCases
                    }
                    unit="条"
                    tone="keep"
                  />
                  <ImpactRow
                    icon={<FileText className="size-3.5" />}
                    label="其它公司知识库文档（仅解除归属，同事仍可检索）"
                    count={impact.kept.crossTenantDocuments}
                    unit="份"
                    tone="keep"
                  />
                </ul>
              </div>
            </div>
          ) : null}

          <div className="space-y-1.5">
            <label
              htmlFor="confirm-company-name"
              className="text-[12px] font-medium text-muted-foreground"
            >
              请输入公司名称「{company.companyName}」以确认
            </label>
            <Input
              id="confirm-company-name"
              value={confirmName}
              onChange={(e) => setConfirmName(e.target.value)}
              placeholder={company.companyName}
              disabled={submitting}
              onKeyDown={(e) => {
                if (e.key === "Enter" && matched) submit();
              }}
            />
          </div>

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              {error}
            </p>
          )}

          <div className="flex gap-2">
            <Button
              variant="outline"
              className="flex-1"
              onClick={onClose}
              disabled={submitting}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              className="flex-1 gap-2"
              onClick={submit}
              // 预检失败/未完成、或未输对公司名时不允许提交：宁可让用户重试，
              // 也不要在不知道影响范围 / 没防误点的情况下点下去
              disabled={submitting || loading || (!impact && !error) || !matched}
            >
              {submitting && <Loader2 className="size-4 animate-spin" />}
              确认删除
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * 删除成员确认弹窗.
 *
 * 打开时先向后端要一份**影响预检**：删掉之后哪些东西真的会消失、哪些会留下，
 * 数字全部来自数据库。产品约定「个人文档消失、部门与公司文档保留」如果只写在
 * 文案里，用户没法验证；放在这里，点确认之前就能看到有几个文件会被带走。
 */
function DeleteMemberDialog({
  member,
  onClose,
  onDone,
}: {
  member: StaffMember | null;
  onClose: () => void;
  onDone: () => Promise<void> | void;
}) {
  const [impact, setImpact] = useState<MemberDeletionImpact | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const memberId = member?.id ?? null;

  useEffect(() => {
    if (!memberId) return;
    let cancelled = false;
    setImpact(null);
    setError(null);
    setLoading(true);
    (async () => {
      try {
        const data = await previewStaffMemberDeletion(memberId);
        if (!cancelled) setImpact(data);
      } catch (err) {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "无法获取删除影响，请稍后重试");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [memberId]);

  if (!member) return null;

  const submit = async () => {
    setError(null);
    setSubmitting(true);
    try {
      const { message } = await deleteStaffMember(member.id);
      toast.success(message);
      await onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && !submitting && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trash2 className="size-4 text-destructive" />
            删除该成员的全部信息？
          </DialogTitle>
          <DialogDescription>
            {member.name}（{member.username}） · {member.companyName}
            {member.departmentName ? ` · ${member.departmentName}` : ""} ·{" "}
            {member.roleLabel}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <p className="flex items-start gap-2 rounded-lg border border-destructive/25 bg-destructive/5 px-3 py-2 text-[12px] leading-relaxed text-destructive">
            <ShieldAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>
              账号与个人数据将被<strong>永久删除且不可恢复</strong>；删除后该成员无法再登录，
              其个人知识库文档、历史对话都会消失。
            </span>
          </p>

          {loading ? (
            <p className="flex items-center gap-2 py-2 text-[12px] text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" />
              正在统计该成员的数据…
            </p>
          ) : impact ? (
            <div className="space-y-3 text-[12px]">
              <div className="rounded-lg border border-border/60 bg-muted/30 px-3 py-2.5">
                <p className="mb-1.5 font-medium text-foreground/90">将被删除</p>
                <ul className="space-y-1">
                  <ImpactRow
                    icon={<FileText className="size-3.5" />}
                    label="个人知识库文档（含原文与向量索引）"
                    count={impact.deleted.personalDocuments}
                    unit="份"
                    tone="danger"
                  />
                  <ImpactRow
                    icon={<MessagesSquare className="size-3.5" />}
                    label="历史对话"
                    count={impact.deleted.conversations}
                    unit="个"
                    tone="danger"
                  />
                  <ImpactRow
                    icon={<MessagesSquare className="size-3.5" />}
                    label="对话消息"
                    count={impact.deleted.messages}
                    unit="条"
                    tone="danger"
                  />
                  <ImpactRow
                    icon={<FileText className="size-3.5" />}
                    label="文档集合（个人分组）"
                    count={impact.deleted.collections}
                    unit="个"
                    tone="danger"
                  />
                </ul>
              </div>

              <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-3 py-2.5">
                <p className="mb-1.5 font-medium text-foreground/90">将被保留</p>
                <ul className="space-y-1">
                  <ImpactRow
                    icon={<FileText className="size-3.5" />}
                    label="部门 / 公司知识库文档（仅解除归属，同事仍可检索）"
                    count={impact.kept.sharedDocuments}
                    unit="份"
                    tone="keep"
                  />
                  <ImpactRow
                    icon={<MessagesSquare className="size-3.5" />}
                    label="身份验证与共享申请的审核留痕"
                    count={impact.kept.staffRequests + impact.kept.shareRequests}
                    unit="条"
                    tone="keep"
                  />
                </ul>
              </div>
            </div>
          ) : null}

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              {error}
            </p>
          )}

          <div className="flex gap-2">
            <Button
              variant="outline"
              className="flex-1"
              onClick={onClose}
              disabled={submitting}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              className="flex-1 gap-2"
              onClick={submit}
              // 预检失败/未完成时不允许提交：宁可让用户重试，也不要在不知道
              // 影响范围的情况下点下去
              disabled={submitting || loading || (!impact && !error)}
            >
              {submitting && <Loader2 className="size-4 animate-spin" />}
              确认删除
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function EditMemberDialog({
  member,
  grantable,
  myRank,
  onClose,
  onDone,
}: {
  member: StaffMember | null;
  grantable: string[];
  myRank: number;
  onClose: () => void;
  onDone: () => Promise<void> | void;
}) {
  const [role, setRole] = useState("");
  const [department, setDepartment] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [active, setActive] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!member) return;
    setRole(member.role);
    setDepartment(member.departmentName ?? "");
    setJobTitle(member.jobTitle ?? "");
    setActive(member.isActive);
    setError(null);
  }, [member]);

  if (!member) return null;

  const roleOptions = Array.from(new Set([member.role, ...grantable])).filter(
    (value) => roleRank(value) < myRank || value === member.role
  );

  const submit = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await updateStaffMember(member.id, {
        role: role !== member.role ? role : undefined,
        departmentName: department.trim() || undefined,
        jobTitle: jobTitle.trim() || undefined,
        isActive: active !== member.isActive ? active : undefined,
      });
      toast.success("职责已更新");
      await onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>更换职责</DialogTitle>
          <DialogDescription>
            {member.name}（{member.username}） · {member.companyName}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[12px] font-medium text-muted-foreground">
              职责角色
            </label>
            <Select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              disabled={submitting}
            >
              {roleOptions.map((value) => (
                <SelectItem key={value} value={value}>
                  {ROLE_LABELS[value] ?? value}
                </SelectItem>
              ))}
            </Select>
            <p className="text-[11px] text-muted-foreground">
              只能授予低于你自己等级的职责；企业管理员全局唯一，不可授予
            </p>
          </div>

          <div className="space-y-1.5">
            <label className="text-[12px] font-medium text-muted-foreground">
              部门
            </label>
            <Input
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              placeholder="留空表示移出部门"
              maxLength={128}
              disabled={submitting}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-[12px] font-medium text-muted-foreground">
              部门职责
            </label>
            <Input
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              placeholder="如：嵌入式软件工程师"
              maxLength={128}
              disabled={submitting}
            />
          </div>

          <label className="flex items-center gap-2 text-[13px]">
            <input
              type="checkbox"
              checked={active}
              onChange={(e) => setActive(e.target.checked)}
              className="size-3.5 accent-primary"
              disabled={submitting}
            />
            账号启用（取消勾选即停用该成员）
          </label>

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              {error}
            </p>
          )}

          <Button className="w-full gap-2" onClick={submit} disabled={submitting}>
            {submitting && <Loader2 className="size-4 animate-spin" />}
            保存修改
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
