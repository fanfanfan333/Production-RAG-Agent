"""Independent synthetic two-hop probe; never replaces the fixed 120-case release set."""
import argparse
import json
from pathlib import Path


def make_challenge():
    parents = "青岚 星瀚 远帆 云杉 北辰 南栖 晨露 锦川 瀚海 松岳 清禾 东篱 望舒 逐风 流萤 岁安 云渡 霁月 观澜 鹿鸣".split()
    subsidiaries = "晨泽 明砾 森途 晴野 泊舟 澄芯 砺行 云衡 沐川 鹤羽 星垣 璞石 秋实 汀兰 沧溟 麦穗 映雪 竹隐 初阳 见山".split()
    parks = "紫荆 红枫 银杏 木棉 白桦 金桂 合欢 玉兰 栀子 海棠 梧桐 云杉 石楠 榕树 木槿 水杉 青槐 山茶 龙柏 侧柏".split()
    documents, cases = {}, []
    for i, parent in enumerate(parents):
        subject = parent + "控股"
        company = subsidiaries[(i * 7 + 3) % 20] + "科技"
        park = parks[(i * 13 + 9) % 20] + "园区"
        first, second = f"relation-{i:02d}-a.md", f"relation-{i:02d}-b.md"
        documents[first] = f"# 投资记录\n{subject}投资{company}。"
        documents[second] = f"# 登记地址\n{company}位于{park}。"
        def case(identifier, category, query, answer, sources):
            return dict(id=identifier, category=category, query=query, answer=answer, expect_sources=sources,
                        source_locations=[dict(filename=name, page_number=1, evidence=documents[name]) for name in sources],
                        expect_refusal=False, expect_conflict_notice=False, human_review="pending_business_review")
        cases.append(case(f"challenge-{i + 1:02d}", "multihop", f"{subject}投资的企业位于哪个园区？", park, [first, second]))
        if i < 4:
            cases.append(case(f"ordinary-{i + 1:02d}", "fact", f"{subject}投资了哪家企业？", company, [first]))
    return dict(name="graph-challenge-v1", license="self-authored",
                truth_origin="explicit synthetic relations; independent fixed permutations, not model answers",
                business_validation="diagnostic subset only; not the 120-case release gate or human-reviewed business data",
                documents=documents, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    golden = make_challenge()
    content = json.dumps(golden, ensure_ascii=False, indent=2) + "\n"
    path = args.output / "gold.json"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        parser.error("输出目录已有不同金标，请使用新目录，不能覆盖已有验收资料")
    (args.output / "documents").mkdir(parents=True, exist_ok=True)
    for name, text in golden["documents"].items():
        (args.output / "documents" / name).write_text(text, encoding="utf-8")
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
