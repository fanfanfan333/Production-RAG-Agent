"""Reviewed examples, deterministic source splits, and TRL-compatible exports."""
import hashlib
import json
import re
import ipaddress
from typing import Literal
from urllib.parse import urlparse, urlunparse, unquote
from pydantic import BaseModel, Field, model_validator


class TrainingEvidence(BaseModel):
    source_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=12000)
    location: str = Field(min_length=1, max_length=256)


class TrainingDraft(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    evidence: list[TrainingEvidence] = Field(min_length=1, max_length=10)
    chosen: str = Field(min_length=1, max_length=20000)
    rejected: str | None = Field(default=None, max_length=20000)
    source_group: str = Field(min_length=1, max_length=256)
    source_kind: Literal["public", "synthetic"]
    license: Literal["CC0", "CC-BY-4.0", "CC-BY-SA-4.0", "self-authored"]

    @model_validator(mode="after")
    def validate_provenance(self):
        if not all(value.strip() for value in (self.prompt, self.chosen, self.source_group,
                   *(value for evidence in self.evidence for value in (evidence.text, evidence.location)))):
            raise ValueError("问题、回答、来源组和证据位置不能为空白")
        if self.rejected is not None and (not self.rejected.strip() or self.rejected.strip() == self.chosen.strip()):
            raise ValueError("偏好答案必须非空且不同，点赞不能替代 chosen/rejected")
        if self.source_kind == "synthetic" and self.license != "self-authored":
            raise ValueError("构造样本必须明确为自有数据")
        for evidence in self.evidence:
            if self.source_kind == "public":
                parsed = urlparse(evidence.source_id)
                hostname = parsed.hostname or ""
                try:
                    private_ip = not ipaddress.ip_address(hostname).is_global
                except ValueError:
                    private_ip = False
                if (parsed.scheme != "https" or not hostname or parsed.username or parsed.password
                        or private_ip or "." not in hostname or hostname.endswith((".localhost", ".local", ".internal"))):
                    raise ValueError("公开来源必须具有可审核的 HTTPS 地址")
                # Group page aliases conservatively; query variants never justify split independence.
                evidence.source_id = urlunparse(("https", hostname.lower(), unquote(parsed.path).rstrip("/") or "/", "", "", ""))
            elif not evidence.source_id.startswith("synthetic:"):
                raise ValueError("构造来源必须使用 synthetic: 标识")
        text = json.dumps(self.model_dump(), ensure_ascii=False)
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|-----BEGIN .*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}", text):
            raise ValueError("样本包含疑似个人信息或密钥，请脱敏并重新审核")
        return self


def export_examples(rows, kind):
    if kind not in {"sft", "dpo"}:
        raise ValueError("unsupported training format")
    drafts = []
    for row in rows:
        if row.status != "approved" or row.reviewed_by is None:
            continue
        draft = TrainingDraft.model_validate({key: getattr(row, key) for key in TrainingDraft.model_fields})
        if kind == "dpo" and draft.rejected is None:
            continue
        drafts.append(draft)
    if not drafts:
        raise ValueError("没有审核通过且符合该格式的数据")
    drafts.sort(key=lambda draft: json.dumps(draft.model_dump(), ensure_ascii=False, sort_keys=True))
    licenses = {}
    for draft in drafts:
        for evidence in draft.evidence:
            if evidence.source_id in licenses and licenses[evidence.source_id] != draft.license:
                raise ValueError("同一来源的使用许可不一致，请重新人工审核")
            licenses[evidence.source_id] = draft.license
    parents = {}
    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            key = parents[key]
        return key
    for draft in drafts:
        keys = [draft.source_group, *(e.source_id for e in draft.evidence)]
        representative = min(root(key) for key in keys)
        for key in keys:
            parents[root(key)] = representative
    train, validation, seen, audit = [], [], {}, []
    for draft in drafts:
        context = "\n\n".join(f"[Source {i + 1}] {e.source_id} {e.location}\n{e.text}" for i, e in enumerate(draft.evidence))
        prompt = [{"role": "system", "content": "只依据提供的证据回答；引用来源，证据不足时明确拒答。"},
                  {"role": "user", "content": context + "\n\n问题：" + draft.prompt}]
        fingerprint = hashlib.sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        component = root(draft.source_group)
        split = "validation" if int(hashlib.sha256(component.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
        sample = {"messages": [*prompt, {"role": "assistant", "content": draft.chosen}]} if kind == "sft" else {
            "prompt": prompt, "chosen": [{"role": "assistant", "content": draft.chosen}],
            "rejected": [{"role": "assistant", "content": draft.rejected}],
        }
        sample_hash = hashlib.sha256(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if fingerprint in seen:
            if seen[fingerprint] != sample_hash:
                raise ValueError("同一问题与证据存在冲突的审核答案，请先人工对账")
            continue
        seen[fingerprint] = sample_hash
        (validation if split == "validation" else train).append(sample)
        audit.append({"sha256": fingerprint, "sample_sha256": sample_hash, "split": split, "sources": [e.source_id for e in draft.evidence],
                      "license": draft.license, "component": component})
    digest = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    return {"format": kind, "dataset_version": digest, "train": train, "validation": validation,
            "manifest": audit, "ready_for_training": bool(train and validation)}
