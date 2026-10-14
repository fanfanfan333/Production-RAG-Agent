"use client";
import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { useAuth } from "@/lib/context/auth-context";
import { apiFetch } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

type Example = {id: string; prompt: string; chosen: string; rejected?: string; status: string; source_kind: string; license: string; evidence: {source_id: string; text: string; location: string}[]};
export default function TrainingPage() {
  const {user} = useAuth();
  const allowed = user?.role === "admin" || user?.permissions?.includes("*");
  const [examples, setExamples] = useState<Example[]>([]);
  const [busy, setBusy] = useState(false);
  const [fields, setFields] = useState({prompt: "", chosen: "", rejected: "", source: "", evidence: "", location: ""});
  const [kind, setKind] = useState("synthetic");
  const [license, setLicense] = useState("self-authored");
  async function refresh() {setExamples(await apiFetch<Example[]>("/training/examples"));}
  useEffect(() => {if (allowed) apiFetch<Example[]>("/training/examples").then(setExamples).catch((error) => toast.error(String(error.message)));}, [allowed]);
  async function create(event: React.FormEvent) {
    event.preventDefault(); setBusy(true);
    try {
      const sourceId = kind === "synthetic" ? `synthetic:${fields.source}` : fields.source;
      await apiFetch("/training/examples", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
        prompt: fields.prompt, chosen: fields.chosen, rejected: fields.rejected || null, source_group: sourceId,
        source_kind: kind, license, evidence: [{source_id: sourceId, text: fields.evidence, location: fields.location}],
      })});
      toast.success("样本已保存，审核通过后才能导出"); await refresh();
    } catch (error) {toast.error(error instanceof Error ? error.message : "保存失败");}
    finally {setBusy(false);}
  }
  async function review(id: string, decision: string) {
    setBusy(true);
    try {await apiFetch(`/training/examples/${id}/review?decision=${decision}`, {method: "POST"}); await refresh();}
    catch (error) {toast.error(error instanceof Error ? error.message : "审核失败");}
    finally {setBusy(false);}
  }
  async function download(format: string) {
    setBusy(true);
    try {
      const result = await apiFetch<{dataset_version: string; ready_for_training: boolean; train: unknown[]; validation: unknown[]}>(`/training/export?kind=${format}`, {method: "POST"});
      const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json"}));
      const link = document.createElement("a"); link.href = url; link.download = `rag-${format}-${result.dataset_version.slice(0, 12)}.json`; link.click(); URL.revokeObjectURL(url);
      toast.success(`已导出：训练 ${result.train.length} 条，验证 ${result.validation.length} 条${result.ready_for_training ? "（尚未训练）" : "；来源分组不足，暂不可训练"}`);
    } catch (error) {toast.error(error instanceof Error ? error.message : "导出失败");}
    finally {setBusy(false);}
  }
  return <AppShell activePath="/admin"><div className="mb-6"><h1 className="text-2xl font-bold">训练样本审核</h1><p className="mt-2 text-sm text-muted-foreground">准备 SFT / DPO 数据，不会自动训练模型。</p></div>
    {!allowed ? <p>仅平台管理员可访问。</p> : <div className="mx-auto max-w-4xl space-y-6">
      <p className="text-sm text-muted-foreground">企业私有文档与普通点赞反馈不会自动进入训练集。请核对原文、纠正答案并确认使用许可；DPO 需要同一问题和证据下的两份不同答案。</p>
      <form onSubmit={create} className="space-y-3 rounded-xl border p-5">
        {Object.entries({prompt: "问题", chosen: "人工纠正后的正确答案", rejected: "较差答案（DPO 必填）", source: kind === "public" ? "公开来源 HTTPS 地址" : "自有构造来源编号", evidence: "原文证据", location: "来源位置（页码或章节）"}).map(([key, label]) => <label key={key} className="block text-sm">{label}<textarea className="mt-1 block w-full rounded-md border bg-background p-2" required={key !== "rejected"} value={fields[key as keyof typeof fields]} onChange={(event) => setFields({...fields, [key]: event.target.value})} maxLength={key === "chosen" || key === "rejected" ? 20000 : key === "evidence" ? 12000 : key === "prompt" ? 4096 : 256} /></label>)}
        <label className="mr-4 inline-flex items-center gap-2 text-sm">数据来源<select className="rounded border bg-background p-1" value={kind} onChange={(event) => {setKind(event.target.value); setLicense(event.target.value === "synthetic" ? "self-authored" : "CC-BY-4.0");}}><option value="synthetic">人工构造</option><option value="public">公开资料</option></select></label>
        <label className="inline-flex items-center gap-2 text-sm">使用许可<select className="rounded border bg-background p-1" value={license} onChange={(event) => setLicense(event.target.value)}>{(kind === "synthetic" ? ["self-authored"] : ["CC0", "CC-BY-4.0", "CC-BY-SA-4.0"]).map((value) => <option key={value}>{value}</option>)}</select></label>
        <Button className="block" disabled={busy}>保存待审核样本</Button>
      </form>
      <div className="flex gap-3"><Button disabled={busy} onClick={() => download("sft")}>导出 SFT 数据</Button><Button disabled={busy} variant="outline" onClick={() => download("dpo")}>导出 DPO 数据</Button></div>
      <div className="space-y-3">{examples.map((example) => <article key={example.id} className="space-y-2 rounded-xl border p-4"><p className="font-medium">{example.prompt}</p><details className="rounded border p-2 text-sm"><summary className="cursor-pointer">核对原文证据与来源</summary>{example.evidence.map((e, i) => <div key={i} className="mt-2"><p className="break-all">{e.source_id} · {e.location}</p><p className="whitespace-pre-wrap">{e.text}</p></div>)}</details><p className="whitespace-pre-wrap text-sm">{example.chosen}</p>{example.rejected && <p className="whitespace-pre-wrap text-sm text-muted-foreground">较差答案：{example.rejected}</p>}<p className="text-xs text-muted-foreground">{example.source_kind} · {example.license} · {example.status === "approved" ? "已审核通过" : example.status === "rejected" ? "已驳回" : "待审核"}</p><div className="flex gap-2"><Button size="sm" disabled={busy} onClick={() => review(example.id, "approved")}>核对原文后通过</Button><Button size="sm" variant="outline" disabled={busy} onClick={() => review(example.id, "rejected")}>驳回</Button></div></article>)}</div>
    </div>}
  </AppShell>;
}
