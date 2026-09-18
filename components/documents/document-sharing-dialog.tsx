"use client";

import { Fragment, useEffect, useState, type ReactNode } from "react";
import {
  ArrowLeft,
  ArrowUpRight,
  Building2,
  CheckCircle2,
  Clock,
  FileX2,
  FolderInput,
  Loader2,
  Lock,
  Send,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { createDeleteRequest, createShareRequest } from "@/lib/api/share";
import {
  getTransferTargets,
  transferDocumentToDepartment,
  updateDocumentVisibility,
  type TransferTargets,
} from "@/lib/api/documents";
import type { AccessLevel, Document, ShareTargetLevel } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { AccessTierBadge, tierScopeName } from "@/components/documents/access-badge";

/**
 * 文档的共享面板：一条路径对应权限矩阵里的一行。
 *
 *   有权限（部门负责人/知识库管理员/企业管理员）
 *       → 直接「发布到部门知识库 / 公司知识库」，或把文档收回个人库
 *   无权限（普通员工）
 *       → 「申请共享」：选择目标层级 + 说明理由，提交给上级审核
 *   已有待审申请
 *       → 显示"审核中"，不再重复提交
 *
 * 界面不使用表情符号，全部用图标 + 克制的色彩表达状态。
 *
 * 组件分两层：外层 `DocumentSharingDialog` 只做 `doc` 判空、自身不持有任何 hook；
 * 内层 `SharingDialogBody` 以非空 `doc` 为入参、承载全部 hook 且无任何提前返回。
 * 之所以这样拆，见外层组件的注释 —— 它是本组件曾经那个 "change in the order of
 * Hooks" 的结构性根治，而不是把某一处 effect 挪个位置。
 */

/** 申请目标的展示元信息（图标 / 名称 / 谁来审）。 */
const TARGET_META = {
  department: { icon: Users, title: "部门知识库", hint: "由本部门负责人审核" },
  tenant: { icon: Building2, title: "公司知识库", hint: "由知识库管理员审核" },
} as const;

/** 弹窗内的首屏面板：`main` 是总览，`transfer` 是「转为部门文档」的二级视图。 */
type SharingPanel = "main" | "transfer";

/**
 * 打开弹窗时希望落地的位置（列表行上的不同入口传入不同的值）。
 *
 *   main      主视图顶部（共享设置入口）
 *   transfer  直达「转为部门文档」的部门选择
 *   request   主视图，并把「申请共享」区提到最前（申请共享入口）
 *   delete    主视图，并把「申请删除」区提到最前（申请删除入口）
 */
export type SharingDialogView = SharingPanel | "request" | "delete";

/** 入口视图 → 首屏面板：只有 transfer 进入二级视图，其余都落在主视图。 */
function panelFor(view: SharingDialogView): SharingPanel {
  return view === "transfer" ? "transfer" : "main";
}

/**
 * 共享弹窗的外层：**判空层**。
 *
 * ⚠️ 本组件刻意**不调用任何 hook**。此前 hooks（9 个 useState + 2 个 useEffect）
 * 与 `if (!doc) return null` 的提前返回交织在同一个组件里：`doc === null` 时渲染
 * 在提前返回处停下（10 个 hook），`doc !== null` 时执行到提前返回之后的第 2 个
 * useEffect（11 个 hook）。父组件用 `open={!!sharingDoc}` 一开一关，hook 数量就
 * 变，于是 React 报 "change in the order of Hooks"。
 *
 * 根治办法是把「hooks 持有者」和「提前返回」从结构上隔离：外层只判空并把非空
 * `doc` 交给内层；内层以非空 `Document` 为入参、无任何提前返回。此后无论内层
 * 再加多少 hook，都不会再随 `doc` 是否为空而变。
 */
export function DocumentSharingDialog({
  doc,
  open,
  onOpenChange,
  onChanged,
  initialView = "main",
}: {
  doc: Document | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChanged?: () => void;
  /** 打开时落在哪个视图，见 {@link SharingDialogView}。 */
  initialView?: SharingDialogView;
}) {
  // 判空只发生在这里，且本组件没有 hook —— 提前返回不会影响任何 hook 的顺序。
  if (!doc) return null;
  return (
    <SharingDialogBody
      // 换文档时强制重挂载：内层无需再处理"同一实例换 doc"的边界状态
      key={doc.id}
      doc={doc}
      open={open}
      onOpenChange={onOpenChange}
      onChanged={onChanged}
      initialView={initialView}
    />
  );
}

/**
 * 共享弹窗的内层：**hooks 持有者**。
 *
 * 契约：`doc` 一定非空（由外层保证），因此本组件内部**没有任何提前返回**，
 * 9 个 useState + 2 个 useEffect 永远在同一个渲染路径上无条件执行。
 */
function SharingDialogBody({
  doc,
  open,
  onOpenChange,
  onChanged,
  initialView,
}: {
  doc: Document;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChanged?: () => void;
  initialView: SharingDialogView;
}) {
  const [target, setTarget] = useState<ShareTargetLevel>("department");
  const [reason, setReason] = useState("");
  const [deleteReason, setDeleteReason] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  // ── 「转为部门文档」是弹窗内的二级视图 ────────────────────────────────────
  //
  // 部门清单按需拉取：只有真的点进这个视图才请求，而不是每次打开共享面板都为
  // 一份文档打一次接口 —— 绝大多数共享动作跟部门归属无关。
  const [view, setView] = useState<SharingPanel>(panelFor(initialView));
  const [targets, setTargets] = useState<TransferTargets | null>(null);
  const [targetDept, setTargetDept] = useState("");
  const [note, setNote] = useState("");
  const [loadingTargets, setLoadingTargets] = useState(false);

  const docId = doc.id;
  useEffect(() => {
    // 每次打开（docId 由 undefined → id）都回到入口指定的视图，并丢掉上一份
    // 文档的部门清单：部门是公司级的，复用看似无害，但"当前归属"高亮会停留在
    // 上一份文档的部门上，看起来像选错了。
    //
    // 外层已用 key={doc.id} 让"换文档"整体重挂载，这一步对首屏是幂等的；保留它
    // 是为了以后万一有人在内层加了提前返回时仍能兜底，同时守住既有行为契约。
    setView(panelFor(initialView));
    setTargets(null);
    setTargetDept("");
    setNote("");
  }, [docId, initialView]);

  const current: AccessLevel = doc.accessLevel ?? "private";
  const pending = Boolean(doc.pendingShareRequest);
  // 能直接删就不用申请；两者互斥（后端能力字段同源，按钮状态与接口判定一致）
  const canRequestDelete = Boolean(doc.canRequestDelete && !doc.canDelete);

  // ── 「直接发布」与「申请共享」是两个**独立**的能力维度，不能合成一个布尔 ────
  //
  // 这里曾经是 `canPublish ? 只渲染直接发布 : 只渲染申请共享`，于是部门负责人
  // （可直接发部门库、不能直接发公司库）整个申请分支被隐藏 —— 界面里只有
  // "发布到部门知识库"，想把文档提到公司库时无路可走。现在按**目标层级**分别
  // 判定，两段可以同时出现在同一个弹窗里。
  const canPublishDepartment =
    Boolean(doc.canPublishDepartment) && current !== "department";
  const canPublishCompany =
    Boolean(doc.canPublishCompany) && current !== "tenant";
  const canRecall = current !== "private";
  const hasAnyPublishRight = Boolean(
    doc.canPublishDepartment || doc.canPublishCompany
  );
  const hasPublishActions =
    canPublishDepartment || canPublishCompany || canRecall;

  // 可申请的层级：没有直接发布权、且还不是该层级的那些
  const requestTargets: ShareTargetLevel[] = [];
  if (doc.canRequestDepartment) requestTargets.push("department");
  if (doc.canRequestCompany) requestTargets.push("tenant");
  const hasRequest = requestTargets.length > 0;
  // target 的初值是 department；该层不可申请时落到第一个可申请的层
  const activeTarget = requestTargets.includes(target)
    ? target
    : requestTargets[0];

  const reset = () => {
    setReason("");
    setDeleteReason("");
    setNote("");
    setBusy(null);
    setView("main");
  };

  const close = (next: boolean) => {
    if (!next) reset();
    onOpenChange(next);
  };

  const publish = async (level: AccessLevel) => {
    setBusy(level);
    try {
      const res = await updateDocumentVisibility(doc.id, level);
      toast.success(res.message || `已更新为${tierScopeName(level)}`);
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "发布失败");
    } finally {
      setBusy(null);
    }
  };

  const submitRequest = async () => {
    if (!activeTarget) return;
    setBusy("request");
    try {
      await createShareRequest({
        documentId: doc.id,
        targetLevel: activeTarget,
        reason: reason.trim() || undefined,
      });
      toast.success("申请已提交，可在「查看申请」跟踪审核进度");
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "提交申请失败");
    } finally {
      setBusy(null);
    }
  };

  const submitDeleteRequest = async () => {
    setBusy("delete");
    try {
      await createDeleteRequest({
        documentId: doc.id,
        reason: deleteReason.trim() || undefined,
      });
      toast.success("删除申请已提交，可在「查看申请」跟踪审核进度");
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "提交删除申请失败");
    } finally {
      setBusy(null);
    }
  };

  // ── 转为部门文档（公司 HR / 知识库管理员）──────────────────────────────────
  //
  // 部门清单**进入这个视图才拉**（按需请求）：绝大多数共享动作与部门归属无关，
  // 没必要每打开一次共享面板都为一份文档打一次接口。拉取失败就退回主视图 ——
  // 停在一个没有部门的空选择页上比报错退出更让人困惑。
  // 依赖里刻意**不放** loadingTargets：它由本 effect 自己置位，放进去会让
  // effect 在 setLoadingTargets(true) 后立刻重跑 —— React 先执行上一次的
  // cleanup（cancelled = true），再让新的一轮早退，于是请求回来时被判定为
  // "已取消"，targets 永远填不上，界面卡在"正在读取公司部门…"。
  // 这里只依赖 view / docId / targets：前者决定要不要拉、后两者决定拉什么。
  // setLoadingTargets / setTargets 是稳定引用，不需要也无法作为依赖。
  useEffect(() => {
    if (view !== "transfer" || !docId || targets) return;
    let cancelled = false;
    setLoadingTargets(true);
    getTransferTargets(docId)
      .then((res) => {
        if (cancelled) return;
        setTargets(res);
        // 默认选中：部门库文档默认它当前的部门，公司库文档默认清单里第一个
        setTargetDept(res.departmentId ?? res.options[0]?.departmentId ?? "");
      })
      .catch((err) => {
        if (cancelled) return;
        toast.error(err instanceof Error ? err.message : "无法获取公司部门清单");
        setView("main");
      })
      .finally(() => {
        if (!cancelled) setLoadingTargets(false);
      });
    return () => {
      cancelled = true;
    };
  }, [view, docId, targets]);

  const openTransfer = () => setView("transfer");

  const submitTransfer = async () => {
    if (!targetDept) return;
    setBusy("transfer");
    try {
      const res = await transferDocumentToDepartment(
        doc.id,
        targetDept,
        note.trim() || undefined
      );
      toast.success(res.message);
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "转换失败");
    } finally {
      setBusy(null);
    }
  };

  // ── 主视图的各动作区（先各自成块，再按入口决定顺序渲染）────────────────────

  // 当前层级
  const currentTierCard = (
    <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3.5 py-3">
      <div>
        <p className="text-xs text-muted-foreground">当前层级</p>
        <p className="mt-1 text-sm font-medium">{tierScopeName(current)}</p>
      </div>
      <AccessTierBadge level={current} label={doc.accessLabel} />
    </div>
  );

  // 待审状态
  const pendingBanner = pending ? (
    <div className="flex items-start gap-2 rounded-lg border border-amber-300/50 bg-amber-50/70 px-3.5 py-3 text-sm text-amber-800 dark:border-amber-700/40 dark:bg-amber-950/30 dark:text-amber-200">
      <Clock className="mt-0.5 size-4 shrink-0" />
      <span>
        这份文档已有一份待审核的共享申请，审核结果可在主界面「查看申请」中查看。
      </span>
    </div>
  ) : null;

  // 有权限的层级：直接发布（与"申请共享"可同时出现）
  const publishSection = hasPublishActions ? (
    <div className="space-y-2.5">
      <p className="text-xs font-medium text-muted-foreground">
        直接发布（你的角色具备发布权限）
      </p>

      {canPublishDepartment && (
        <TierAction
          icon={Users}
          title="发布到部门知识库"
          description="同部门的成员按权限即可检索到这份文档"
          loading={busy === "department"}
          disabled={busy !== null}
          onClick={() => publish("department")}
        />
      )}

      {canPublishCompany && (
        <TierAction
          icon={Building2}
          title="发布到公司知识库"
          description="全公司成员按权限即可检索到这份文档"
          loading={busy === "tenant"}
          disabled={busy !== null}
          onClick={() => publish("tenant")}
        />
      )}

      {canRecall && (
        <TierAction
          icon={Lock}
          title="收回至个人知识库"
          description="仅你本人可见，已共享的成员将无法再检索到"
          loading={busy === "private"}
          disabled={busy !== null}
          onClick={() => publish("private")}
        />
      )}
    </div>
  ) : null;

  // 公司级管理者：把已共享的文档**改归到指定部门**
  // 与上面的「发布到部门知识库」不是一回事：那条发的是"我自己的部门"，
  // 这条由操作者在公司已有部门里挑目标（HR 把薪酬制度下沉给人力部）。
  const transferSection = Boolean(doc.canTransferDepartment) ? (
    <div className="space-y-2.5">
      <p className="text-xs font-medium text-muted-foreground">调整部门归属</p>
      <TierAction
        icon={FolderInput}
        title="转为部门文档"
        description={
          current === "tenant"
            ? "选择一个部门，把公司文档转为该部门所属的部门文档"
            : "把这份部门文档改归到公司里的另一个部门"
        }
        loading={false}
        disabled={busy !== null}
        onClick={openTransfer}
      />
    </div>
  ) : null;

  // 无直接发布权的层级：申请共享
  const requestSection = hasRequest ? (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-muted-foreground">申请共享到</p>
        <div
          className={cn(
            "grid gap-2",
            requestTargets.length > 1 ? "grid-cols-2" : "grid-cols-1"
          )}
        >
          {requestTargets.map((level) => {
            const meta = TARGET_META[level];
            return (
              <TargetOption
                key={level}
                active={activeTarget === level}
                icon={meta.icon}
                title={meta.title}
                hint={meta.hint}
                onClick={() => setTarget(level)}
              />
            );
          })}
        </div>
      </div>

      {pending ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          这份文档已有一份待审核的申请（一次只提交一份，避免审核队列被
          同一份文档刷屏）。审核通过后会自动发布到目标知识库，无需再次
          操作。
        </p>
      ) : (
        <>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              申请说明（可选）
            </label>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="简要说明为什么需要共享，便于审核人判断"
              rows={3}
              maxLength={1000}
            />
          </div>

          <p className="text-xs leading-relaxed text-muted-foreground">
            {hasAnyPublishRight
              ? "提交后由上一级权限的管理员审核（部门库 → 本部门负责人；公司库 → 知识库管理员 / 企业管理员），审核结果会出现在「查看申请」中。"
              : doc.publishDeniedReason ||
                "提交后由上一级权限的管理员审核，审核结果会出现在「查看申请」中。"}
          </p>

          <Button
            className="w-full gap-2"
            disabled={busy !== null}
            onClick={submitRequest}
          >
            {busy === "request" ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Send className="size-4" />
            )}
            提交共享申请
          </Button>
        </>
      )}
    </div>
  ) : null;

  // 没有删除权但看得见该文档 → 申请删除，由上级同意或拒绝
  // 已有待审申请时整块隐藏：此时后端会以 409 拒绝新的删除申请，界面不该给
  // 用户一个注定失败的按钮。
  const deleteSection =
    canRequestDelete && !pending ? (
      <div className="space-y-3 rounded-lg border border-border/60 bg-muted/20 px-3.5 py-3">
        <div className="flex items-start gap-2">
          <FileX2 className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          <div className="space-y-1">
            <p className="text-sm font-medium">申请删除</p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              {doc.deleteDeniedReason ||
                `${tierScopeName(current)}的文档由上级统一删除。你可以提交申请说明理由，由 ${
                  current === "tenant" ? "知识库管理员或企业管理员" : "部门负责人"
                } 审核。`}
            </p>
          </div>
        </div>

        <Textarea
          value={deleteReason}
          onChange={(e) => setDeleteReason(e.target.value)}
          placeholder="简要说明为什么需要删除这份文档，便于审核人判断"
          rows={3}
          maxLength={1000}
        />

        <Button
          variant="outline"
          className="w-full gap-2"
          disabled={busy !== null}
          onClick={submitDeleteRequest}
        >
          {busy === "delete" ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <FileX2 className="size-4" />
          )}
          提交删除申请
        </Button>
      </div>
    ) : null;

  // 既不能发布、也没有可申请的层级，且不是已有待审申请
  const emptySection =
    !hasPublishActions && !hasRequest && !pending ? (
      <p className="text-xs text-muted-foreground">
        {doc.publishDeniedReason ||
          "这份文档已在你可操作的最高层级，无需再调整。"}
      </p>
    ) : null;

  // ── 入口落地：把入口语义对应的动作区提到最前 ────────────────────────────────
  //
  // 列表行上的「申请共享 / 申请删除」按钮语义是"去提这个申请"，但主视图里它们
  // 排在「直接发布 / 调整部门归属」之后，用户点完还得往下找，入口语义与落地位置
  // 不一致。这里按入口把对应的动作区提到最前（申请段置顶），其余入口（共享设置）
  // 保持常规顺序（发布在前），不改变管理路径的观感。
  //
  // 说明：弹窗是 `position: fixed`，scrollIntoView 无法可靠地滚动 fixed 子树，
  // 所以这里用"置顶"而不是"滚动到可见"来实现"直达"。
  const focusSection: "request" | "delete" | null =
    initialView === "request"
      ? "request"
      : initialView === "delete"
        ? "delete"
        : null;

  const actionSections: { key: string; node: ReactNode }[] = [];
  if (publishSection) actionSections.push({ key: "publish", node: publishSection });
  if (transferSection) actionSections.push({ key: "transfer", node: transferSection });
  if (requestSection) actionSections.push({ key: "request", node: requestSection });
  if (deleteSection) actionSections.push({ key: "delete", node: deleteSection });
  if (emptySection) actionSections.push({ key: "empty", node: emptySection });

  const orderedActions =
    focusSection === null
      ? actionSections
      : [
          ...actionSections.filter((section) => section.key === focusSection),
          ...actionSections.filter((section) => section.key !== focusSection),
        ];

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="truncate pr-6">{doc.name}</DialogTitle>
          <DialogDescription>
            {view === "transfer"
              ? "选择这份文档要归属的部门；转换后只有该部门的成员能检索到它"
              : "调整这份文档所在的知识库层级，或向上级申请共享 / 申请删除"}
          </DialogDescription>
        </DialogHeader>

        {view === "transfer" ? (
          <TransferDepartmentPanel
            targets={targets}
            loading={loadingTargets}
            value={targetDept}
            onChange={setTargetDept}
            note={note}
            onNoteChange={setNote}
            busy={busy === "transfer"}
            disabled={busy !== null}
            onBack={() => setView("main")}
            onConfirm={submitTransfer}
          />
        ) : (
          <div className="space-y-5">
            {currentTierCard}
            {pendingBanner}
            {orderedActions.map((section) => (
              <Fragment key={section.key}>{section.node}</Fragment>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function TierAction({
  icon: Icon,
  title,
  description,
  loading,
  disabled,
  onClick,
}: {
  icon: typeof Users;
  title: string;
  description: string;
  loading: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border border-border/60 px-3.5 py-3 text-left transition-colors",
        "hover:border-border hover:bg-muted/50",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted">
        {loading ? (
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
        ) : (
          <Icon className="size-4 text-muted-foreground" />
        )}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium">{title}</span>
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {description}
        </span>
      </span>
      <ArrowUpRight className="size-4 shrink-0 text-muted-foreground" />
    </button>
  );
}

function TransferDepartmentPanel({
  targets,
  loading,
  value,
  onChange,
  note,
  onNoteChange,
  busy,
  disabled,
  onBack,
  onConfirm,
}: {
  targets: TransferTargets | null;
  loading: boolean;
  value: string;
  onChange: (departmentId: string) => void;
  note: string;
  onNoteChange: (note: string) => void;
  busy: boolean;
  disabled: boolean;
  onBack: () => void;
  onConfirm: () => void;
}) {
  const options = targets?.options ?? [];
  const currentDept = targets?.departmentId ?? null;
  const selected = options.find((o) => o.departmentId === value);

  return (
    <div className="space-y-4">
      <button
        type="button"
        onClick={onBack}
        disabled={disabled}
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
      >
        <ArrowLeft className="size-3.5" />
        返回
      </button>

      <div className="space-y-1">
        <p className="text-sm font-medium">转为部门文档</p>
        <p className="text-xs leading-relaxed text-muted-foreground">
          选择这份文档要归属的部门。转换后只有该部门的成员能检索到它，其他部门的
          同事将无法再看到 —— 这是把一份组织资产收窄到具体部门的动作。
        </p>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          正在读取公司已有部门…
        </div>
      ) : options.length === 0 ? (
        <p className="rounded-lg border border-border/60 bg-muted/20 px-3.5 py-3 text-xs leading-relaxed text-muted-foreground">
          公司还没有任何有成员归属的部门。请先在「成员管理」里为同事设置部门，
          再回来把文档转为部门文档。
        </p>
      ) : (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-muted-foreground">
            公司已有部门（{options.length}）
          </p>
          <div className="max-h-60 space-y-1.5 overflow-y-auto pr-1">
            {options.map((opt) => {
              const active = opt.departmentId === value;
              const isCurrent = opt.departmentId === currentDept;
              return (
                <button
                  key={opt.departmentId}
                  type="button"
                  disabled={disabled}
                  onClick={() => onChange(opt.departmentId)}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition-colors",
                    active
                      ? "border-primary/60 bg-primary/5"
                      : "border-border/60 hover:bg-muted/50",
                    disabled && "cursor-not-allowed opacity-60"
                  )}
                >
                  <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted">
                    <Users className="size-4 text-muted-foreground" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">
                      {opt.departmentName}
                    </span>
                    <span className="mt-0.5 block text-[11px] text-muted-foreground">
                      {opt.memberCount} 名成员
                      {isCurrent ? " · 当前归属" : ""}
                    </span>
                  </span>
                  {active && (
                    <CheckCircle2 className="size-4 shrink-0 text-primary" />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {options.length > 0 && (
        <>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              转换说明（可选，会写入审计日志）
            </label>
            <Textarea
              value={note}
              onChange={(e) => onNoteChange(e.target.value)}
              rows={2}
              maxLength={200}
              placeholder="例如：薪酬制度仅限人力资源部查阅"
            />
          </div>

          <p className="rounded-lg border border-border/60 bg-muted/20 px-3.5 py-3 text-xs leading-relaxed text-muted-foreground">
            转换后其他部门的同事将无法再检索到这份文档。如需恢复全公司可见，可在
            共享面板里再点「发布到公司知识库」。
          </p>

          <Button
            className="w-full gap-2"
            disabled={disabled || loading || !value}
            onClick={onConfirm}
          >
            {busy ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <FolderInput className="size-4" />
            )}
            {selected ? `确认转为「${selected.departmentName}」的文档` : "确认转换"}
          </Button>
        </>
      )}
    </div>
  );
}

function TargetOption({
  active,
  icon: Icon,
  title,
  hint,
  onClick,
}: {
  active: boolean;
  icon: typeof Users;
  title: string;
  hint: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors",
        active
          ? "border-primary/60 bg-primary/5"
          : "border-border/60 hover:bg-muted/50"
      )}
    >
      <span className="flex items-center gap-1.5 text-sm font-medium">
        <Icon className="size-3.5" />
        {title}
        {active && <CheckCircle2 className="size-3.5 text-primary" />}
      </span>
      <span className="text-[11px] text-muted-foreground">{hint}</span>
    </button>
  );
}
