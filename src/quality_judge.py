"""
质量判定器（Gate-1 / Gate-2）
==============================

对 MinerU 提取后的 Markdown 做质量判定，实现「能识别低质量数据并自动分流」。

设计原则：规则层 + 模型层双轨
--------------------------------
- **规则层**：处理可量化的硬指标（乱码率、公式还原率、表格召回率）。
  快速、确定、可复现、零成本，适合拦住「明显不合格」。
- **模型层**：处理语义问题（内容截断、章节断层、占位符残留）。
  结构完整但内容空洞的文档，规则拦不住，需要语义理解。

判定逻辑
--------
    通过 = 规则层全部通过 AND 模型层评分 >= 阈值

本模块实现规则层与模型层接口；模型层通过 `QualityJudge.llm_score()`
调用，缺省可用启发式近似（`heuristic=True`），便于离线运行与单测。

用法
----
    from quality_judge import QualityJudge, GateConfig
    judge = QualityJudge(GateConfig(threshold=60))
    ok, reason, detail = judge.judge(markdown, profile)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- 正则

CID_RE = re.compile(r"\(cid:\s*\d+\)")
LATEX_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
LATEX_INLINE_RE = re.compile(r"(?<!\$)\$([^$\n]{2,120})\$(?!\$)")
MD_TABLE_RE = re.compile(r"^\s*\|.+?\|\s*$", re.M)
MD_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$", re.M)
PAGE_NUM_RE = re.compile(r"^\s*(?:-\s*)?\d+\s*(?:/\s*\d+)?\s*(?:-\s*)?$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.M)

PLACEHOLDER_PATTERNS = (
    re.compile(r"待补充|待完善|待填|此处省略"),
    re.compile(r"\bTODO\b|\bFIXME\b|\bXXX+\b"),
    re.compile(r"\[插入图\]|\[表格\]|\[图\s*\d*\]"),
    re.compile(r"\.{6,}\s*$"),
)

HEADER_FOOTER_HINTS = (
    "preprint", "working paper", "submitted", "under review",
    "doi:", "doi.org", "arxiv", "copyright", "©", "all rights reserved",
    "issn", "isbn", "https://", "http://",
)


# ---------------------------------------------------------------- 配置


@dataclass
class GateConfig:
    """质量闸门配置。"""

    # —— 规则层阈值 ——
    max_cid_ratio: float = 0.40        # CID 乱码率上限
    min_latex_ratio: float = 0.60      # 公式还原率下限
    min_table_recall: float = 1.00     # 表格召回率下限
    max_residual_noise: int = 5        # 残留页眉/页码上限

    # —— 模型层阈值 ——
    threshold: int = 60                # 综合评分阈值
    min_readability: int = 15          # 可读性维度最低分

    def to_dict(self) -> dict:
        return {
            "rules": {
                "max_cid_ratio": self.max_cid_ratio,
                "min_latex_ratio": self.min_latex_ratio,
                "min_table_recall": self.min_table_recall,
                "max_residual_noise": self.max_residual_noise,
            },
            "model": {
                "threshold": self.threshold,
                "min_readability": self.min_readability,
            },
        }


# ---------------------------------------------------------------- 指标计算


def count_cid(text: str) -> int:
    return len(CID_RE.findall(text))


def count_latex(text: str) -> int:
    """统计 LaTeX 公式块数（块级 $$ + 行内 $）。"""
    blocks = LATEX_BLOCK_RE.findall(text)
    # 去掉已计入块级的部分，避免重复计数
    rest = LATEX_BLOCK_RE.sub("", text)
    inlines = LATEX_INLINE_RE.findall(rest)
    return len(blocks) + len(inlines)


def count_md_tables(text: str) -> int:
    """统计 Markdown 表格数（按表头分隔行计数）。"""
    return len(MD_TABLE_SEP_RE.findall(text))


def count_residual_noise(text: str) -> int:
    """统计残留的页码与页眉页脚噪声。"""
    n = 0
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if PAGE_NUM_RE.match(s):
            n += 1
            continue
        low = s.lower()
        if len(s) < 120 and any(h in low for h in HEADER_FOOTER_HINTS):
            n += 1
    return n


def count_placeholders(text: str) -> int:
    n = 0
    for p in PLACEHOLDER_PATTERNS:
        n += len(p.findall(text))
    return n


def heading_levels(text: str) -> list[int]:
    return [len(m.group(1)) for m in HEADING_RE.finditer(text)]


# 句子终结符（含中英文标点与右括号、右引号）
TERMINATORS = set(
    "。．.！!？?：:；;）)】」』〕"
    "”’\"'"  # 右双引号、右单引号、直双引号、直单引号
)


def has_truncation(text: str) -> bool:
    """判断文档是否被截断（末句不完整）。"""
    tail = text.rstrip()
    if not tail:
        return True
    # 结尾没有任何终结标点，且不是代码块/表格/列表
    last_line = tail.split("\n")[-1].strip()
    if not last_line:
        return True
    if last_line.startswith(("|", "```", "$", "-", "*", ">")):
        return False
    return last_line[-1] not in TERMINATORS


# ---------------------------------------------------------------- 判定器


@dataclass
class JudgeResult:
    """判定结果。"""

    passed: bool
    reason: str
    score: int
    rule_passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    dimension_scores: dict[str, int] = field(default_factory=dict)
    verdict: str = "reject"

    def to_json(self) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "reason": self.reason,
                "score": self.score,
                "rule_passed": self.rule_passed,
                "dimension_scores": self.dimension_scores,
                "verdict": self.verdict,
                "detail": self.detail,
            },
            ensure_ascii=False,
            indent=2,
        )


class QualityJudge:
    """质量判定器：规则层 + 模型层。"""

    def __init__(self, config: GateConfig | None = None,
                 use_llm: bool = False) -> None:
        self.cfg = config or GateConfig()
        self.use_llm = use_llm

    # -------------------------------------------------- 规则层

    def judge_rules(self, md: str, profile: dict | None) -> tuple[bool, str, dict]:
        """规则层判定。

        Parameters
        ----------
        md : str
            MinerU 输出的 Markdown。
        profile : dict | None
            PDF 预分析结果（含公式数、表格数等基线）。
        """
        detail: dict[str, Any] = {}
        n_cid = count_cid(md)
        n_latex = count_latex(md)
        n_tbl = count_md_tables(md)
        noise = count_residual_noise(md)

        detail.update(n_cid=n_cid, n_latex=n_latex,
                      n_md_tables=n_tbl, residual_noise=noise)

        # 规则1：CID 乱码率
        n_lines = max(len([x for x in md.split("\n") if x.strip()]), 1)
        cid_ratio = n_cid / n_lines
        detail["cid_ratio"] = round(cid_ratio, 4)
        if cid_ratio > self.cfg.max_cid_ratio:
            return False, f"cid_garbage_high({cid_ratio:.1%}>{self.cfg.max_cid_ratio:.0%})", detail

        # 规则2：公式还原率（需基线）
        if profile:
            expected = len(profile.get("total_eq_numbers", []) or [])
            if expected > 0:
                ratio = n_latex / expected
                detail["latex_ratio"] = round(ratio, 4)
                detail["expected_formula"] = expected
                if ratio < self.cfg.min_latex_ratio:
                    return (False,
                            f"formula_loss({n_latex}/{expected}={ratio:.1%}"
                            f"<{self.cfg.min_latex_ratio:.0%})", detail)

            # 规则3：表格召回率
            exp_tbl = profile.get("total_tables", 0) or 0
            if exp_tbl > 0:
                recall = min(n_tbl / exp_tbl, 1.0)
                detail["table_recall"] = round(recall, 4)
                detail["expected_table"] = exp_tbl
                if recall < self.cfg.min_table_recall - 1e-9:
                    return (False,
                            f"table_loss({n_tbl}/{exp_tbl}={recall:.1%}"
                            f"<{self.cfg.min_table_recall:.0%})", detail)

        # 规则4：残留噪声
        if noise > self.cfg.max_residual_noise:
            return False, f"residual_noise({noise}>{self.cfg.max_residual_noise})", detail

        return True, "rules_passed", detail

    # -------------------------------------------------- 模型层

    def llm_score(self, md: str) -> dict:
        """调用 LLM 评分（需自行接入模型）。

        正式实验应调用小模型并传入 `config/prompts/quality_judge.txt`。
        此处提供启发式近似，保证流水线可离线运行。
        """
        if self.use_llm:
            # —— 此处接入真实模型 ——
            # prompt = Path("config/prompts/quality_judge.txt").read_text()
            # prompt = prompt.replace("{{markdown_content}}", md[:8000])
            # raw = call_llm(prompt)
            # return json.loads(raw)
            pass

        return self._heuristic_score(md)

    @staticmethod
    def _heuristic_score(md: str) -> dict:
        """启发式评分：与 quality_judge.txt 的四个维度对齐。"""
        n_chars = len(md)
        n_lines = max(len([x for x in md.split("\n") if x.strip()]), 1)
        n_cid = count_cid(md)
        placeholders = count_placeholders(md)
        levels = heading_levels(md)
        truncated = has_truncation(md)

        # 1) 完整性 25分：过短或截断扣分
        completeness = 25
        if n_chars < 500:
            completeness = 5
        elif n_chars < 2000:
            completeness = 15
        if truncated:
            completeness = max(completeness - 10, 0)

        # 2) 结构连贯 25分：无标题、层级跳跃扣分
        coherence = 25
        if not levels:
            coherence = 8
        else:
            jumps = sum(1 for a, b in zip(levels, levels[1:]) if b - a > 1)
            coherence = max(coherence - jumps * 3, 5)

        # 3) 实质信息量 25分：占位符密集扣分
        density = 25
        ph_rate = placeholders / n_lines
        if ph_rate > 0.05:
            density = max(int(25 * (1 - ph_rate * 10)), 0)

        # 4) 可读性 25分：乱码率主导
        readability = 25
        cid_rate = n_cid / n_lines
        if cid_rate > 0.02:
            readability = max(int(25 * (1 - cid_rate * 25)), 0)
        readability = max(readability - min(placeholders, 5), 0)

        return {
            "score": completeness + coherence + density + readability,
            "dimension_scores": {
                "completeness": completeness,
                "coherence": coherence,
                "information_density": density,
                "readability": readability,
            },
            "verdict": "accept",
            "reason": "heuristic",
        }

    # -------------------------------------------------- 综合判定

    def judge(self, md: str, profile: dict | None = None) -> JudgeResult:
        """综合判定：规则层 AND 模型层。"""
        rule_ok, rule_reason, detail = self.judge_rules(md, profile)

        llm = self.llm_score(md)
        dims = llm.get("dimension_scores", {})
        score = int(llm.get("score", 0))
        readability = int(dims.get("readability", 0))

        model_ok = (score >= self.cfg.threshold
                    and readability >= self.cfg.min_readability)

        passed = rule_ok and model_ok

        if not rule_ok:
            reason = rule_reason
        elif not model_ok:
            reason = (f"low_score({score}<{self.cfg.threshold})"
                      if score < self.cfg.threshold
                      else f"low_readability({readability}<{self.cfg.min_readability})")
        else:
            reason = "passed"

        detail["llm_score"] = score
        detail["dimension_scores"] = dims

        return JudgeResult(
            passed=passed,
            reason=reason,
            score=score,
            rule_passed=rule_ok,
            detail=detail,
            dimension_scores=dims,
            verdict="accept" if passed else "reject",
        )
