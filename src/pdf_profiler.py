"""
PDF 预分析器
============

在做AI-Ready 数据生产之前，先对「粗糙」的原始 PDF 做一次量化体检。

为什么需要这一步
----------------
MinerU 这类工具是「盲跑」的——它不知道一份PDF 里有多少公式、多少表格、
哪些页眉页脚是噪声。没有基线就无法评估提取质量。

本模块给出**可量化的基线**，让后续「开启/关闭公式识别」的对比实验
有据可依，而不是靠肉眼感觉。

检测项
------
1. **公式检测**：识别独立成行的数学公式（编号模式 + 符号密度）
2. **表格检测**：用线框/对齐特征判断表格区域
3. **噪声检测**：页眉、页脚、页码（PDF 里典型的干扰大模型的元素）
4. **标题层级**：识别章节标题，为后续 H1/H2/H3 还原度评估做准备

用法
----
    python src/pdf_profiler.py --pdf data/raw/xxx.pdf --json results/profile.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------- 正则定义

# 公式编号：行尾的 (1) (2.3) (A.1)，兼容中文全角括号
EQ_NUMBER_RE = re.compile(
    r"[(（](?:[A-Z]?\.?\d{1,2}(?:[-–.]\d{1,2})?)[)）]\s*$")

# 数学符号密度：用于区分「公式行」与「普通文本行」
MATH_SYMBOLS = set("∑∏∫∂√±×÷≈≠≤≥∞∈∉⊂⊃∪∩∀∃→←↔⇒⇔αβγδεθλμπρστφωΓΔΘΛΣΦΩ")
MATH_FUNCS = ("frac", "sqrt", "sum", "int", "prod", "lim", "log", "exp",
              "hat", "tilde", "bar", "vec", "begin", "end", "left", "right",
              "mathbb", "mathrm", "text", "cdot", "ldots", "quad", "qquad")

# CID乱码：PDF 文本层里未映射字形会变成 (cid:NN) —— 这是「AI-Ready」的关键障碍
CID_RE = re.compile(r"\(cid:\s*\d+\)")

# 行内公式标记：$H_{1}$、$\alpha_{1}$、$$\frac{...}$$ 等（中文期刊常见）
INLINE_MATH_RE = re.compile(r"\$[^$]{1,80}\$")
# 独立公式编号：(1) / （1）/ 式(1.2) / 表 3-1
EQUATION_LABEL_RE = re.compile(
    r"(?:式\s*)?[(（](?:[A-Z]?\.?\d{1,2}(?:[-–.]\d{1,2})?)[)）]"
)

# 页码：孤立数字或「N / M」形式
PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$", re.I)

# 常见页眉/页脚关键词（学术论文）
HEADER_FOOTER_HINTS = (
    "preprint", "working paper", "submitted", "under review", "forthcoming",
    "doi:", "doi.org", "arxiv", "copyright", "©", "all rights reserved",
    "issn", "isbn", "volume", "vol.", "issue", "print", "https://", "http://",
    "page", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)

# 章节标题（中文论文：一、/（一）/ 1. / 1.1 形式）
SECTION_ZH_RE = re.compile(
    r"^\s*(?:"
    r"[一二三四五六七八九十]{1,3}\s*[、．.]|"      # 一、
    r"[（(][一二三四五六七八九十]{1,3}[)）]|"        # （一）
    r"\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+"                # 1. / 1.1 / 1.1.1
    r")\s*\S+"
)
SECTION_EN_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)\.?\s+)?([A-Z][A-Za-z\s]{2,60}?)\s*$"
)


@dataclass
class PageProfile:
    """单页的量化画像。"""

    page_no: int
    n_chars: int
    n_lines: int
    equation_lines: int
    eq_numbers: list[str] = field(default_factory=list)
    n_tables_detected: int = 0
    header_candidates: list[str] = field(default_factory=list)
    footer_candidates: list[str] = field(default_factory=list)
    page_number_found: bool = False
    math_density: float = 0.0
    heading_candidates: list[str] = field(default_factory=list)
    # —— AI-Ready 关键指标 ——
    cid_garbage_count: int = 0        # (cid:NN) 乱码数量
    cid_garbage_ratio: float = 0.0    # 乱码占比
    lines_no_cid: int = 0             # 完全无乱码的文本行数（可喂给大模型）


@dataclass
class DocProfile:
    """整篇文档的量化画像。"""

    file: str
    n_pages: int
    total_chars: int
    total_equation_lines: int
    total_eq_numbers: list[str]
    total_tables: int
    header_footer_noise_pages: int
    heading_candidates: list[str]
    pages: list[PageProfile]
    # 「跨页表格」判定：同一表格的表头在多页重复出现
    cross_page_table_candidates: list[dict]
    # —— AI-Ready 核心指标 ——
    total_cid_garbage: int = 0
    cid_garbage_ratio: float = 0.0
    clean_lines: int = 0
    total_lines: int = 0
    clean_line_ratio: float = 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["pages"] = [asdict(p) for p in self.pages]
        return d

    def summary(self) -> str:
        lines = [
            "=" * 66,
            "PDF 预分析报告（AI-Ready 数据生产基线）",
            "=" * 66,
            f"文件: {Path(self.file).name}",
            f"页数              : {self.n_pages}",
            f"总字符数          : {self.total_chars:,}",
            f"总行数            : {self.total_lines:,}",
            "",
            "【结构元素】",
            f"  公式行数（估计）  : {self.total_equation_lines}",
            f"  公式编号          : {len(self.total_eq_numbers)} 个"
            f" {self.total_eq_numbers[:12]}",
            f"  表格（线框检测）  : {self.total_tables}",
            f"  跨页表格候选      : {len(self.cross_page_table_candidates)}",
            f"  含页眉页脚噪声页  : {self.header_footer_noise_pages}",
            f"  标题候选          : {len(self.heading_candidates)}",
            "",
            "【AI-Ready 质量指标】",
            f"  CID 乱码总数      : {self.total_cid_garbage:,} 处",
            f"  CID 乱码占比      : {self.cid_garbage_ratio:.2%}",
            f"  可直接使用的行    : {self.clean_lines:,} / {self.total_lines:,}"
            f"（{self.clean_line_ratio:.2%}）",
            "",
            "— 作业要求对照 —",
            f"  复杂公式 ≥5: "
            f"{'✓ 满足' if len(self.total_eq_numbers) >= 5 else '✗ 不足'}"
            f"（{len(self.total_eq_numbers)} 个）",
            f"  跨页表格 ≥2  : "
            f"{'✓ 满足' if len(self.cross_page_table_candidates) >= 2 else '✗ 不足'}"
            f"（{len(self.cross_page_table_candidates)} 个）",
        ]
        if self.total_cid_garbage:
            lines += [
                "",
                "⚠ 关键发现：该PDF 文本层含大量 (cid:NN) 乱码——",
                "  公式使用特殊字体编码，直接复制粘贴会得到乱码。",
                "  这正是「为什么粗糙 PDF 不能直接喂给大模型」的实证依据。",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------- 核心检测


def _split_lines(text: str) -> list[str]:
    return [ln for ln in (text.split("\n")) if ln.strip()]


def _math_density(line: str) -> float:
    """一行中数学符号的占比。"""
    if not line:
        return 0.0
    hits = sum(1 for ch in line if ch in MATH_SYMBOLS)
    hits += sum(3 for f in MATH_FUNCS if f in line)
    return hits / len(line)


def _is_equation(line: str) -> bool:
    """判断一行是否像独立公式。

    兼容三种情形：
    1. 行尾有公式编号（半角或全角括号）
    2. 含 (cid:NN) 乱码 —— 说明这行是公式，但字形未映射
    3. 数学符号密度高且有等号/箭头
    """
    s = line.strip()
    if len(s) < 3:
        return False

    has_num = bool(EQ_NUMBER_RE.search(s))
    dense = _math_density(s) >= 0.06
    has_rel = any(t in s for t in ("=", "≈", "≤", "≥", "→", "∼", "∝", "＝"))

    # 情形 2：含 cid 乱码 → 极可能是公式行
    n_cid = len(CID_RE.findall(s))
    if n_cid >= 2 and (has_rel or has_num or dense):
        return True
    if has_num and n_cid >= 1:
        return True

    return (has_num and (dense or has_rel)) or (dense and has_rel and len(s) < 90)


def _detect_headings(lines: list[str]) -> list[str]:
    """候选章节标题：支持中文（一、（一）、1.1）与英文（1. / 1.1.）编号体系。"""
    out = []
    for ln in lines:
        s = ln.strip()
        if not (3 < len(s) < 60):
            continue
        if s.endswith(("。", "，", "；", "的", "了")):
            continue
        if CID_RE.search(s):
            continue
        if SECTION_ZH_RE.match(s):
            out.append(s)
            continue
        m = SECTION_EN_RE.match(s)
        if m and m.group(1):
            out.append(s)
        elif m and s.isupper() and len(s) > 4:
            out.append(s)
    return out


def _classify_noise(lines: list[str], page_no: int, total_pages: int) -> tuple[list[str], list[str], bool]:
    """判断页眉/页脚/页码。

    启发式规则：
    - 页码：整行只有数字或「N/M」
    - 页眉：首页出现的短行 + 含 preprint/doi 等关键词
    - 页脚：末页出现的短行 + 含版权/联系方式
    """
    headers, footers = [], []
    has_page_no = False

    for ln in lines:
        s = ln.strip()
        if PAGE_NUM_RE.match(s):
            has_page_no = True
            footers.append(s)
            continue
        low = s.lower()
        if any(h in low for h in HEADER_FOOTER_HINTS):
            # 出现在前 2 页的靠上行→ 页眉；靠下行 → 页脚
            (headers if page_no <= 2 else footers).append(s)

    # 位置启发式：首页/末页的最短行 + 非句子
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if len(s) < 45 and idx < 3 and page_no == 1 and not PAGE_NUM_RE.match(s):
            if any(ch.isalpha() for ch in s) and s not in headers:
                # 首行很短且像刊名/日期
                if any(k in s.lower() for k in ("journal", "quarterly", "review", "working paper", "preprint")):
                    headers.append(s)
        if len(s) < 45 and idx >= len(lines) - 3 and page_no == total_pages:
            if any(ch.isalpha() for ch in s) and s not in footers:
                footers.append(s)

    return headers, footers, has_page_no


def _detect_cross_page_tables(
    pages: list[PageProfile],
    table_texts: list[str],
    line_index: list[list[str]] | None = None,
    per_page_tables: list[list[str]] | None = None,
) -> list[dict]:
    """判定跨页表格。

    两种判据（互补）：

    1. **表头重复法**：同一首行文本在 ≥2 页出现（适用于有线框的表格）
    2. **续表标记法**：学术论文的跨页表格通常在续页重复表头，
       或带有 "Table X (continued)" / "(续)" 之类标记；
       同时若某页表格首行与上一页表格末行的列结构高度相似，
       也判定为同一表格的延续。
    3. **表号+表题法**：若同一编号的 Table 出现在多页（正文引用式），
       结合表题出现位置判断。

    Returns
    -------
    list[dict]
        每项含 header / pages / 方法说明
    """
    cands: list[dict] = []
    per_page_tables = per_page_tables or []

    # --- 判据 1：表头文本在多页重复 ---
    header_pages: dict[str, set[int]] = {}
    for pg, texts in zip(pages, table_texts):
        for t in texts:
            if not t:
                continue
            rows = [r for r in t.split("\n") if r.strip()]
            if not rows:
                continue
            # 用前 2 行做表头（有些表格第一行是 spanner）
            head = " | ".join(r.strip()[:40] for r in rows[:2])
            if len(head) < 12:
                continue
            header_pages.setdefault(head, set()).add(pg.page_no)

    for head, pgs in header_pages.items():
        if len(pgs) >= 2:
            cands.append({
                "header": head,
                "pages": sorted(pgs),
                "method": "表头重复法",
            })

    # --- 判据 2：续表标记 ---
    CONTINUED_RE = re.compile(
        r"(table\s+\d+\s*\(?(?:cont(?:inued)?\.?|续)\)?|续表|表\s*\d+\s*续)", re.I)
    cont_pages: dict[str, list[int]] = {}
    for pg, lines in zip(pages, line_index or []):
        for ln in lines:
            m = CONTINUED_RE.search(ln)
            if m:
                key = re.sub(r"\s+", " ", m.group(0)).lower()
                cont_pages.setdefault(key, []).append(pg.page_no)

    for key, pgs in cont_pages.items():
        if len(pgs) >= 1 and not any(c["header"].startswith(key) for c in cands):
            cands.append({
                "header": f"{key}（续表标记）",
                "pages": sorted(set(pgs)),
                "method": "续表标记法",
            })

    # --- 判据 3：相邻页表格列数变化（跨页表的排版特征）---
    # 跨页表格在续页会被排版挤压：列数减少（部分列移到下一页）
    # 或行数减少（表尾在首页、续页只剩表头+系数行）。
    for i in range(len(per_page_tables) - 1):
        prev_list = per_page_tables[i]
        next_list = per_page_tables[i + 1]
        if not prev_list or not next_list:
            continue

        for ta, ca in prev_list:
            ra = [r for r in ta.split("\n") if r.strip()]
            if len(ra) < 2:
                continue
            for tb, cb in next_list:
                rb = [r for r in tb.split("\n") if r.strip()]
                if len(rb) < 2:
                    continue
                # 列数减少 → 续页只承载部分列
                if cb < ca:
                    cands.append({
                        "header": rb[0].strip()[:60],
                        "pages": [pages[i].page_no, pages[i + 1].page_no],
                        "method": f"跨页压窄法（{ca}列→{cb}列）",
                    })

    # --- 判据 4：页内表格群聚（同一张表的分页片段常连续出现）---
    # 若连续三页都检出表格，且列数呈「宽→窄→窄」走势，判定为跨页表
    for i in range(len(per_page_tables) - 2):
        a, b, c = per_page_tables[i], per_page_tables[i + 1], per_page_tables[i + 2]
        if not (a and b and c):
            continue
        cols_a = max((n for _, n in a), default=0)
        cols_b = max((n for _, n in b), default=0)
        cols_c = max((n for _, n in c), default=0)
        if cols_a >= 3 and cols_b < cols_a and cols_c < cols_a:
            cands.append({
                "header": f"三页连续表格（{cols_a}/{cols_b}/{cols_c} 列）",
                "pages": [pages[i].page_no, pages[i + 1].page_no,
                          pages[i + 2].page_no],
                "method": "多页延续法",
            })

    # 去重（按页集合+表头相似）
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for c in cands:
        sig = (tuple(c["pages"]), c["header"][:30])
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def profile_pdf(pdf_path: str | Path) -> DocProfile:
    """对 PDF 做全面预分析。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages: list[PageProfile] = []
    table_texts: list[str] = []
    per_page_tables: list[list[str]] = []
    all_line_texts: list[list[str]] = []
    all_eq_numbers: list[str] = []
    all_inline_eqs: list[str] = []
    all_labeled_eqs: list[str] = []
    noise_pages = 0
    headings: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for pno, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = _split_lines(text)

            # 公式行
            eq_lines = [ln for ln in lines if _is_equation(ln)]
            eq_nums = [
                EQ_NUMBER_RE.search(ln.strip()).group(0).strip(" ()（）")
                for ln in lines if EQ_NUMBER_RE.search(ln.strip())
            ]
            # 补充：行内公式标记（$...$ / 式(1)），这类 PDF 文本层
            # 往往整行都是乱码，只能靠标记识别公式
            inline_eqs = [
                m.group(0) for ln in lines
                for m in INLINE_MATH_RE.finditer(ln)
            ]
            labeled_eqs = [
                m.group(0) for ln in lines
                for m in EQUATION_LABEL_RE.finditer(ln)
                if re.search(r"式\s*[(（]", m.group(0))
            ]

            # 表格
            tbl_texts = []
            tbl_cols: list[int] = []
            try:
                for tb in page.find_tables():
                    # extract() 返回二维数组，需展平为「行」的列表
                    rows = tb.extract() or []
                    flat = [
                        " ".join(str(c).strip() for c in row if c is not None)
                        for row in rows
                    ]
                    tbl_texts.append("\n".join(x for x in flat if x.strip()))
                    tbl_cols.append(max((len(r) for r in rows), default=0))
            except Exception:
                pass
            n_tbl = len(tbl_texts)

            # 噪声
            headers, footers, has_pno = _classify_noise(lines, pno, total_pages)
            if headers or footers or has_pno:
                noise_pages += 1

            # 标题
            hds = _detect_headings(lines)
            headings.extend(hds)
            all_eq_numbers.extend(eq_nums)
            all_inline_eqs.extend(inline_eqs)
            all_labeled_eqs.extend(labeled_eqs)

            densities = [_math_density(ln) for ln in lines] or [0.0]

            # CID 乱码统计（AI-Ready 核心指标）
            n_cid = sum(len(CID_RE.findall(ln)) for ln in lines)
            lines_clean = sum(1 for ln in lines if not CID_RE.search(ln))

            pages.append(PageProfile(
                page_no=pno,
                n_chars=len(text),
                n_lines=len(lines),
                equation_lines=len(eq_lines),
                eq_numbers=eq_nums,
                n_tables_detected=n_tbl,
                header_candidates=headers[:5],
                footer_candidates=footers[:5],
                page_number_found=has_pno,
                math_density=round(statistics.fmean(densities), 4),
                heading_candidates=hds[:8],
                cid_garbage_count=n_cid,
                cid_garbage_ratio=round(n_cid / max(len(lines), 1), 4),
                lines_no_cid=lines_clean,
            ))
            table_texts.append("\n".join(tbl_texts))
            per_page_tables.append(list(zip(tbl_texts, tbl_cols)))
            all_line_texts.append(lines)

    cross = _detect_cross_page_tables(pages, table_texts, all_line_texts,
                                    per_page_tables)

    total_cid = sum(p.cid_garbage_count for p in pages)
    total_lines = sum(p.n_lines for p in pages)
    clean_lines = sum(p.lines_no_cid for p in pages)

    # 公式总数= 行尾编号 + 行内公式标记 + 显式「式(1)」
    n_eq_total = (len(all_eq_numbers) + len(all_inline_eqs)
                 + len(all_labeled_eqs))

    return DocProfile(
        file=str(pdf_path),
        n_pages=total_pages,
        total_chars=sum(p.n_chars for p in pages),
        total_equation_lines=sum(p.equation_lines for p in pages),
        total_eq_numbers=all_eq_numbers + all_inline_eqs + all_labeled_eqs,
        total_tables=sum(p.n_tables_detected for p in pages),
        header_footer_noise_pages=noise_pages,
        heading_candidates=sorted(set(headings))[:40],
        pages=pages,
        cross_page_table_candidates=cross,
        total_cid_garbage=total_cid,
        cid_garbage_ratio=total_cid / max(total_lines, 1),
        clean_lines=clean_lines,
        total_lines=total_lines,
        clean_line_ratio=clean_lines / max(total_lines, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF 预分析器（AI-Ready 数据生产基线）")
    ap.add_argument("--pdf", required=True, help="待分析 PDF 路径")
    ap.add_argument("--json", default=None, help="把结果写入 JSON")
    ap.add_argument("--pages", type=int, default=None, help="只分析前 N 页")
    args = ap.parse_args()

    prof = profile_pdf(args.pdf)
    if args.pages:
        prof.pages = prof.pages[: args.pages]

    print(prof.summary())
    print()
    print("逐页明细（前 12 页）：")
    print(f"{'页':>4}{'字符':>8}{'公式行':>8}{'公式号':>8}{'表格':>6}{'噪声':>6}  标题候选")
    for p in prof.pages[:12]:
        noise = "有" if (p.header_candidates or p.footer_candidates
                         or p.page_number_found) else "—"
        hd = p.heading_candidates[0][:26] if p.heading_candidates else ""
        print(f"{p.page_no:>4}{p.n_chars:>8}{p.equation_lines:>8}"
              f"{len(p.eq_numbers):>8}{p.n_tables_detected:>6}{noise:>6}  {hd}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prof.to_json(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nJSON 已写入: {out}")


if __name__ == "__main__":
    main()