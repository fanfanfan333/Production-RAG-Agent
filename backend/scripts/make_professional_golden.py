"""Author deterministic fixture facts, not LLM-generated reference answers."""
import argparse
import json
from pathlib import Path


def make_golden():
    documents, cases = {}, []
    def add(category, question, answer, sources, *, refused=False, conflict=False):
        cases.append({"id": f"{category}-{sum(c['category'] == category for c in cases) + 1:02d}", "category": category,
                      "query": question, "answer": answer, "expect_sources": sources,
                      "scope": {"tenant": "fixture-alpha", "owner": "fixture-editor"},
                      "expect_refusal": refused, "expect_conflict_notice": conflict,
                      "review": "fixture_fact_verified"})
        cases[-1]["human_review"] = "pending_business_review"
        cases[-1]["source_locations"] = [{"filename": source, "page_number": 1, "evidence": documents[source]} for source in sources]
    for i in range(20):
        name = f"fact-{i:02d}.md"
        code, power, days = f"ABX-{i + 100}", 1000 + i * 25, 30 + i
        documents[name] = f"# {code} 产品手册\n额定功率为{power}瓦。维护周期为{days}天。"
        add("fact", f"{code}的额定功率是多少？", f"{power}瓦", [name])
        add("fact", f"{code}每隔多少天维护？", f"{days}天", [name])
    for i in range(5):
        name = f"table-{i:02d}.md"
        values = [(f"部门{j + 1}", 100 + i * 100 + j * 7) for j in range(5)]
        documents[name] = f"# 202{ i }预算\n| 部门 | 预算（万元） |\n|---|---|\n" + "\n".join(f"| {department} | {value} |" for department, value in values)
        for department, value in values:
            add("table", f"202{i}年{department}预算是多少？", f"{value}万元", [name])
    for i in range(15):
        name = f"image-{i:02d}.png"
        documents[name] = f"图表{i + 1}\n项目代号 VIZ-{i + 100}\n收入 {500 + i * 31} 万元\n统计年份 2025"
        add("image", f"图表{i + 1}的项目收入是多少？", f"{500 + i * 31}万元", [name])
    for i in range(20):
        left, right = f"relation-{i:02d}-a.md", f"relation-{i:02d}-b.md"
        documents[left] = f"# 企业投资\n企业甲{i}投资企业乙{i}。"
        documents[right] = f"# 企业地址\n企业乙{i}位于园区{i}。"
        add("multihop", f"企业甲{i}投资的企业位于哪个园区？", f"园区{i}", [left, right])
    for i in range(10):
        add("negative", f"不存在的设备 MISSING-{i}保修几年？", None, [], refused=True)
    for i in range(10):
        left, right = f"conflict-{i:02d}-a.md", f"conflict-{i:02d}-b.md"
        documents[left] = f"# 未注明日期的记录甲\n项目冲突{i}预算为{100 + i}万元。"
        documents[right] = f"# 未注明日期的记录乙\n项目冲突{i}预算为{200 + i}万元。"
        add("negative", f"项目冲突{i}预算究竟是多少？", "材料冲突，无法确定", [left, right], conflict=True)
    return {"name": "professional-fixture-v1", "license": "self-authored", "truth_origin": "explicit fixture facts and arithmetic; not model answers",
            "business_validation": "not a substitute for reviewed enterprise documents", "documents": documents, "cases": cases}


def verify(golden):
    expected = {"fact": 40, "table": 25, "image": 15, "multihop": 20, "negative": 20}
    assert {key: sum(c["category"] == key for c in golden["cases"]) for key in expected} == expected
    assert len({c["id"] for c in golden["cases"]}) == 120
    for case in golden["cases"]:
        assert all(name in golden["documents"] for name in case["expect_sources"])
        if not case["expect_refusal"] and not case["expect_conflict_notice"]:
            evidence = "".join(golden["documents"][name] for name in case["expect_sources"])
            assert case["answer"].replace("万元", "").replace(" ", "") in evidence.replace(" ", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("eval/professional_v1.json"))
    parser.add_argument("--documents", type=Path, help="Optional local fixture documents directory")
    args = parser.parse_args()
    golden = make_golden()
    verify(golden)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.documents:
        args.documents.mkdir(parents=True, exist_ok=True)
        for name, text in golden["documents"].items():
            if name.endswith(".png"):
                from PIL import Image, ImageDraw, ImageFont
                image = Image.new("RGB", (1200, 700), "white")
                font_paths = ["C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
                font_path = next((path for path in font_paths if Path(path).exists()), None)
                if font_path is None:
                    raise RuntimeError("需要中文字体，不能生成缺字图片夹具")
                draw = ImageDraw.Draw(image)
                draw.multiline_text((60, 80), text, fill="black", font=ImageFont.truetype(font_path, 44), spacing=30)
                image.save(args.documents / name)
            else:
                (args.documents / name).write_text(text, encoding="utf-8")
    print(f"Verified {len(golden['cases'])} deterministic fixture cases")


if __name__ == "__main__":
    main()
