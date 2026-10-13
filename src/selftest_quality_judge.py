"""
质量判定器自测脚本
==================

用三个典型样本验证「自动分流」能力：
1. 粗糙 PDF 文本层（含大量 CID 乱码）→ 应被 Gate-1 拦下
2. 结构完整的 Markdown→ 应通过
3. 含占位符/截断的文档 → 应被 Gate-2 拦下

用法
----
    python src/selftest_quality_judge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quality_judge import GateConfig, QualityJudge, count_latex, count_md_tables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    judge = QualityJudge(GateConfig())

    # 样本 2/3 使用本仓库的流水线设计文档作为「结构良好的 Markdown」
    good_md = (ROOT / "docs" / "pipeline_design.md").read_text(encoding="utf-8")

    # 样本 1：粗糙 PDF 文本层
    raw = ""
    pdf_path = ROOT / "data" / "raw" / "Li_Huang_2016_SOEs_Performance.pdf"
    if pdf_path.exists():
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            raw = "\n".join((p.extract_text() or "") for p in pdf.pages)

    # 样本 3：占位符残留 + 截断
    placeholder = (
        "# 研究设计\n\n待补充\n\n"
        + "这是正文内容。" * 80
        + "\n\n[插入图]\nTODO"
    )

    # 与各样本实际内容匹配的基线
    prof_good = {
        "total_eq_numbers": ["(1)"] * max(count_latex(good_md), 1),
        "total_tables": max(count_md_tables(good_md), 1),
    }
    prof_raw = {"total_eq_numbers": ["(1)"] * 63, "total_tables": 6}

    cases = [
        ("粗糙PDF文本层", raw, prof_raw, False),
        ("结构化Markdown", good_md, prof_good, True),
        ("占位符残留", placeholder, prof_good, False),
    ]

    print("=" * 66)
    print("质量判定器自测（Gate-1 / Gate-2）")
    print("=" * 66)

    all_ok = True
    for name, text, prof, expect_pass in cases:
        res = judge.judge(text, prof)
        ok = res.passed == expect_pass
        all_ok &= ok
        mark = "✓" if ok else "✗"
        print(f"\n{mark} {name}")
        print(f"    预期通过={expect_pass}  实际通过={res.passed}  评分={res.score}")
        print(f"    拒绝理由: {res.reason}")
        print(f"    维度得分: {res.dimension_scores}")

    print("\n" + "=" * 66)
    print(f"自测结果: {'全部通过' if all_ok else '存在失败'}")
    print("=" * 66)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
