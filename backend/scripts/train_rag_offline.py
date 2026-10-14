"""Validate an audited export by default; explicitly execute LoRA only on a separate GPU."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_bundle(bundle):
    kind = bundle.get("format")
    if kind not in {"sft", "dpo"} or not bundle.get("train") or not bundle.get("validation"):
        raise ValueError("需要非空且按来源隔离的训练/验证集")
    manifest = bundle.get("manifest", [])
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if version != bundle.get("dataset_version"):
        raise ValueError("数据版本校验失败")
    source_sets = {"train": set(), "validation": set()}
    components = {"train": set(), "validation": set()}
    for split in source_sets:
        samples = bundle[split]
        records = [row for row in manifest if row.get("split") == split]
        if len(samples) != len(records):
            raise ValueError("样本与来源清单数量不符")
        for sample, record in zip(samples, records):
            digest = hashlib.sha256(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if digest != record.get("sample_sha256"):
                raise ValueError("样本内容被修改，需重新审核导出")
            if kind == "sft":
                if not sample.get("messages") or sample["messages"][-1].get("role") != "assistant":
                    raise ValueError("无有效 SFT 回答")
            elif not sample.get("prompt") or not sample.get("chosen") or not sample.get("rejected") or sample["chosen"] == sample["rejected"]:
                raise ValueError("无有效 DPO 偏好对")
            source_sets[split].update(record.get("sources", []))
            if not record.get("sources") or not record.get("component"):
                raise ValueError("缺少来源或来源组")
            components[split].add(record["component"])
    if source_sets["train"] & source_sets["validation"] or components["train"] & components["validation"]:
        raise ValueError("训练集与验证集存在来源泄漏")
    return kind


def training_samples(bundle, tokenizer, max_length=2048):
    """Reject truncation before loading weights; SFT loss covers the answer only."""
    result = {}
    for split in ("train", "validation"):
        result[split] = []
        for sample in bundle[split]:
            if bundle["format"] == "sft":
                item = {"prompt": sample["messages"][:-1], "completion": sample["messages"][-1:]}
                sequences = [sample["messages"]]
            else:
                item = sample
                sequences = [sample["prompt"] + sample[key] for key in ("chosen", "rejected")]
            prompt_messages = item["prompt"]
            prompt_tokens = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
            full_tokens = [tokenizer.apply_chat_template(sequence, tokenize=True) for sequence in sequences]
            if any(tokens[:len(prompt_tokens)] != prompt_tokens or len(tokens) <= len(prompt_tokens) for tokens in full_tokens):
                raise ValueError("对话模板无法正确分离问题与答案，拒绝训练")
            if any(len(tokens) + 1 > max_length for tokens in full_tokens):
                raise ValueError("样本超过2048 token，拒绝静默截断；请缩短证据并重新审核")
            result[split].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, help="Local Hugging Face model directory; Ollama GGUF is not a training model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Opt in to actual training on a separate GPU host")
    args = parser.parse_args()
    bundle = json.loads(args.dataset.read_text(encoding="utf-8"))
    kind = validate_bundle(bundle)
    print(json.dumps({"validated": True, "format": kind, "dataset_version": bundle["dataset_version"],
                      "train": len(bundle["train"]), "validation": len(bundle["validation"])}))
    if not args.execute:
        return
    if not args.model or not (args.model / "config.json").is_file() or not args.output:
        parser.error("执行训练需要本地 HF 模型目录和独立输出目录")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空，拒绝覆盖已有模型")
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        parser.error("本机不运行重训练；请使用至少16GB可用显存的独立训练环境")
    if not torch.cuda.is_bf16_supported():
        parser.error("训练环境不支持 BF16")
    weights = list(args.model.glob("*.safetensors")) or list(args.model.glob("pytorch_model*.bin"))
    if not weights or sum(path.stat().st_size for path in weights) > torch.cuda.mem_get_info()[0] // 3:
        parser.error("模型权重过大或缺失；需为训练激活值保留显存，禁止卸载到主机内存")
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        parser.error("模型缺少对话模板")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    samples = training_samples(bundle, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, device_map={"": 0})
    config_type, trainer_type = (SFTConfig, SFTTrainer) if kind == "sft" else (DPOConfig, DPOTrainer)
    config = config_type(output_dir=str(args.output), per_device_train_batch_size=1, gradient_accumulation_steps=8,
                         num_train_epochs=1, learning_rate=2e-5, max_length=2048, bf16=True,
                         gradient_checkpointing=True, report_to="none", save_strategy="epoch",
                         **({"completion_only_loss": True} if kind == "sft" else {}))
    trainer = trainer_type(model=model, args=config, processing_class=tokenizer,
                           train_dataset=Dataset.from_list(samples["train"]), eval_dataset=Dataset.from_list(samples["validation"]),
                           peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(str(args.output))
    (args.output / "dataset-version.json").write_text(json.dumps({"version": bundle["dataset_version"], "format": kind,
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "training_config": config.to_dict(), "base_model": str(args.model.resolve()), "peft_rank": 8}, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
