"""
第一阶段对比实验：公式/表格识别开/关的影响
============================================

作业要求：
    「尝试对比开启/关闭"公式识别"或"表格识别"功能后的输出差异，
      观察效果变化」

实验设计
--------
四组对照（2×2 因子设计）：

| 组| formula_enable | table_enable | 用途 |
|----|----------------|---------------|------|
| A  | ✓              | ✓             | 全功能（基准）|
| B  | ✗              | ✓             | 只关公式 |
| C  | ✓              | ✗             | 只关表格 |
| D  | ✗              | ✗             | 全关闭（对照）|

评价指标（均以PDF 预分析为基线计算）
------------------------------------
- 公式还原率   = 提取出的 LaTeX 块数 / 预分析检出的公式数
- 表格召回率   = 提取出的表格数 / 预分析检出的表格数
- 标题层级还原 = Markdown 标题数 / 预分析检出的标题候选数
- 噪声残留     = 页码 + 页眉页脚关键词命中数
- 有效字符占比 = 无(cid:NN) 乱码的行占比
- 综合质量评分 = quality_judge 的四维打分

用法
----
    python src/run_comparison.py --pdf data/raw/paper.pdf
    python src/run_comparison.py --pdf data/raw/paper.pdf --baseline results/profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extraction import ExtractConfig, ExtractResult, MinerUExtractor  # noqa: E402
from pdf_profiler import DocProfile, profile_pdf  # noqa: E402
from quality_judge import (  # noqa: E402
    GateConfig, QualityJudge,
    count_cid, count_latex, count_md_tables,
    count_residual_noise, heading_levels,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 实验组定义


@dataclass
class Variant:
    """一个实验组。"""

    key: str
    name: str
    formula_enable: bool
    table_enable: bool


VARIANTS = [
    Variant("A", "全功能（基准）", True, True),
    Variant("B", "关闭公式识别", False, True),
    Variant("C", "关闭表格识别", True, False),
    Variant("D", "公式+表格全关", False, False),
]


@dataclass
class Metrics:
    """一组实验的量化结果。"""

    variant: str
    name: str
    formula_enable: bool
    table_enable: bool
    ok: bool

    n_chars: int = 0
    n_latex: int = 0
    n_tables: int = 0
    n_cid: int = 0
    residual_noise: int = 0
    n_headings: int = 0
    heading_levels_seen: str = ""

    # 相对基线的比率
    formula_recall: float = 0.0
    table_recall: float = 0.0
    heading_recall: float = 0.0
    clean_line_ratio: float = 0.0
    quality_score: int = 0
    quality_dims: dict = None
    rejected_reason: str = ""
    elapsed_sec: float = 0.0
    error: str = ""


# ---------------------------------------------------------------- 基线


def load_baseline(pdf: Path, baseline_json: Path | None) -> dict:
    """获取预分析基线：优先读JSON，否则现场计算。"""
    if baseline_json and baseline_json.exists():
        return json.loads(baseline_json.read_text(encoding="utf-8"))

    print("[基线] 未提供 JSON，正在现场计算 PDF 预分析…")
    prof: DocProfile = profile_pdf(pdf)
    return prof.to_json()


def baseline_numbers(baseline: dict) -> dict:
    """从基线中抽出用于对比的关键数字。"""
    return {
        "n_formula": len(baseline.get("total_eq_numbers", []) or []),
        "n_table": baseline.get("total_tables", 0) or 0,
        "n_heading": len(baseline.get("heading_candidates", []) or []),
        "n_pages": baseline.get("n_pages", 0) or 0,
        "cross_page_tables": len(
            baseline.get("cross_page_table_candidates", []) or []),
    }


# ---------------------------------------------------------------- 评估


def evaluate(res: ExtractResult, baseline: dict,
             judge: QualityJudge) -> Metrics:
    """对一次提取结果计算全部指标。"""
    md = res.markdown
    bl = baseline_numbers(baseline)

    n_latex = count_latex(md)
    n_tables = count_md_tables(md)
    n_cid = count_cid(md)
    noise = count_residual_noise(md)
    levels = heading_levels(md)
    n_lines = max(len([x for x in md.split("\n") if x.strip()]), 1)
    clean_ratio = 1 - (n_cid / n_lines)

    formula_recall = n_latex / bl["n_formula"] if bl["n_formula"] else float("nan")
    table_recall = min(n_tables / bl["n_table"], 1.0) if bl["n_table"] else float("nan")
    heading_recall = (
        min(len(levels) / bl["n_heading"], 1.0) if bl["n_heading"] else float("nan"))

    jr = judge.judge(md, baseline)

    return Metrics(
        variant="",
        name="",
        formula_enable=False,
        table_enable=False,
        ok=res.ok,
        n_chars=res.n_chars,
        n_latex=n_latex,
        n_tables=n_tables,
        n_cid=n_cid,
        residual_noise=noise,
        n_headings=len(levels),
        heading_levels_seen="".join(str(x) for x in sorted(set(levels))),
        formula_recall=round(formula_recall, 4) if formula_recall == formula_recall else -1,
        table_recall=round(table_recall, 4) if table_recall == table_recall else -1,
        heading_recall=round(heading_recall, 4) if heading_recall == heading_recall else -1,
        clean_line_ratio=round(clean_ratio, 4),
        quality_score=jr.score,
        quality_dims=jr.dimension_scores,
        rejected_reason="" if jr.passed else jr.reason,
        elapsed_sec=round(res.elapsed_sec, 2),
        error=res.error,
    )


# -------------------------------------------------- 表格化输出


def _fmt(v: float) -> str:
    if v is None or v < 0:
        return "—"
    return f"{v:.1%}"


def print_table(metrics: list[Metrics], bl: dict) -> None:
    """打印对比表（三线表体例）。"""
    w = 66
    print("\n" + "=" * w)
    print("表1  公式/表格识别开关对提取质量的影响")
    print("=" * w)
    print(f"素材：{bl['n_pages']} 页 | 基线检出：公式 {bl['n_formula']} 个、"
          f"表格 {bl['n_table']} 个（其中跨页 {bl['cross_page_tables']} 个）、"
          f"标题 {bl['n_heading']} 个")
    print("-" * w)
    print(f"{'组':<3}{'配置':<16}{'公式还原':>10}{'表格召回':>10}"
          f"{'标题还原':>10}{'有效字符':>10}{'质量分':>8}")
    print("-" * w)
    for m in metrics:
        cfg = ("公式✓表格✓" if m.formula_enable and m.table_enable
               else "公式✗表格✓" if m.table_enable
               else "公式✓表格✗" if m.formula_enable
               else "公式✗表格✗")
        print(f"{m.variant:<3}{cfg:<16}{_fmt(m.formula_recall):>10}"
              f"{_fmt(m.table_recall):>10}{_fmt(m.heading_recall):>10}"
              f"{_fmt(m.clean_line_ratio):>10}{m.quality_score:>8}")
    print("-" * w)
    print("注：各列为相对 PDF 预分析基线的比率；「有效字符」= 无(cid:NN) 乱码的行占比。")
    print("=" * w)


def print_detail(metrics: list[Metrics]) -> None:
    """打印每组的详细指标。"""
    print("\n" + "=" * 66)
    print("逐组明细")
    print("=" * 66)
    for m in metrics:
        print(f"\n【{m.variant}组】{m.name}")
        if not m.ok:
            print(f"  提取失败: {m.error}")
            continue
        print(f"  输出字符数    : {m.n_chars:,}")
        print(f"  LaTeX 公式块  : {m.n_latex}")
        print(f"  Markdown 表格 : {m.n_tables}")
        print(f"  标题层级      : {m.n_headings} 个，出现层级 {m.heading_levels_seen}")
        print(f"  CID 乱码      : {m.n_cid} 处")
        print(f"  噪声残留      : {m.residual_noise} 处")
        print(f"  质量评分      : {m.quality_score}/100  {m.quality_dims}")
        print(f"  闸门判定      : {'通过' if not m.rejected_reason else '拦下 — ' + m.rejected_reason}")
        print(f"  耗时          : {m.elapsed_sec}s")


# ---------------------------------------------------------------- 主流程


def run(pdf: Path, baseline_json: Path | None,
        out_dir: Path) -> tuple[list[Metrics], dict]:
    """执行四组对比实验。"""
    baseline = load_baseline(pdf, baseline_json)
    bl = baseline_numbers(baseline)

    from extraction import availability
    avail = availability()
    if not avail["usable"]:
        print("！MinerU 未安装，无法执行真实提取。")
        print("  请先运行：pip install -U 'mineru[core]'")
        print("  本次仅输出基线，供离线核对。")
        return [], baseline

    print(f"MinerU 状态：API={avail['api_available']} "
          f"CLI={avail['cli_available']}")
    print(f"\n开始 {len(VARIANTS)} 组对照实验…\n")

    judge = QualityJudge(GateConfig())
    metrics: list[Metrics] = []

    for v in VARIANTS:
        print(f"[{v.key}] {v.name} …", end="", flush=True)
        cfg = ExtractConfig(
            formula_enable=v.formula_enable,
            table_enable=v.table_enable,
            output_dir=ROOT / "data" / "processed" / f"variant_{v.key}",
        )
        ex = MinerUExtractor(cfg)
        res = ex.extract(pdf)
        m = evaluate(res, baseline, judge)
        m.variant, m.name = v.key, v.name
        m.formula_enable, m.table_enable = v.formula_enable, v.table_enable
        metrics.append(m)

        # 落盘产物
        if res.ok:
            vdir = ROOT / "data" / "processed" / f"variant_{v.key}"
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"{pdf.stem}.md").write_text(res.markdown,
                                                 encoding="utf-8")
            (vdir / "metrics.json").write_text(
                json.dumps(asdict(m), ensure_ascii=False, indent=2),
                encoding="utf-8")
        print("完成" if res.ok else f"失败（{res.error[:60]}）")

    return metrics, baseline


def main() -> None:
    ap = argparse.ArgumentParser(
        description="公式/表格识别开/关的对比实验")
    ap.add_argument("--pdf", required=True, help="实验用PDF")
    ap.add_argument("--baseline", default=None,
                    help="预分析 JSON（缺省则现场计算）")
    ap.add_argument("--out", default="results", help="输出目录")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"文件不存在: {pdf}")
        sys.exit(1)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, baseline = run(pdf, Path(args.baseline) if args.baseline else None,
                            out_dir)

    if not metrics:
        # MinerU 不可用时，至少把基线写出来
        (out_dir / "baseline.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n基线已保存: {out_dir / 'baseline.json'}")
        return

    print_table(metrics, baseline_numbers(baseline))
    print_detail(metrics)

    payload = {
        "pdf": str(pdf),
        "baseline": baseline_numbers(baseline),
        "variants": [asdict(m) for m in metrics],
    }
    (out_dir / "comparison_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_dir / 'comparison_table.json'}")


if __name__ == "__main__":
    main()
