"""
MinerU 提取封装（支持公式/表格识别开关）
=========================================

对 MinerU 的薄封装，让「开启/关闭公式识别、表格识别」成为可配置的参数，
从而支撑第一阶段的对比实验。

设计要点
--------
1. **开关可配置**：formula_enable / table_enable 是实验的自变量
2. **优雅降级**：MinerU 未安装时给出明确提示，不让流水线崩溃
3. **可复现**：固定随机种子，记录版本与配置到产物元数据

MinerU 的两种调用方式
---------------------
- **库方式**（推荐）：`from mineru.cli.common import do_parse`
- **命令行**：`mineru -p <pdf> -o <outdir>`

本模块优先用库方式；不可用时回退到命令行。

用法
----
    from extraction import MinerUExtractor, ExtractConfig

    cfg = ExtractConfig(formula_enable=True, table_enable=True)
    ex = MinerUExtractor(cfg)
    result = ex.extract("data/raw/paper.pdf")
    print(result.markdown[:500])
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractConfig:
    """提取配置。"""

    formula_enable: bool = True
    table_enable: bool = True
    backend: str = "pipeline"          # pipeline | vlm
    lang: str = "ch"# ch | en
    device: str = "cpu"
    output_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir) if self.output_dir else None
        return d


@dataclass
class ExtractResult:
    """提取结果。"""

    ok: bool
    markdown: str = ""
    n_chars: int = 0
    n_latex: int = 0
    n_tables: int = 0
    n_cid: int = 0
    headings: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "n_chars": self.n_chars,
                "n_latex": self.n_latex,
                "n_tables": self.n_tables,
                "n_cid": self.n_cid,
                "headings": self.headings[:20],
                "elapsed_sec": round(self.elapsed_sec, 2),
                "error": self.error,
                "meta": self.meta,
            },
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------- 指标


def _count_latex(text: str) -> int:
    import re
    blocks = re.findall(r"\$\$(.+?)\$\$", text, re.S)
    rest = re.sub(r"\$\$(.+?)\$\$", "", text, flags=re.S)
    inlines = re.findall(r"(?<!\$)\$([^$\n]{2,120})\$(?!\$)", rest)
    return len(blocks) + len(inlines)


def _count_tables(text: str) -> int:
    import re
    return len(re.findall(r"^\s*\|[\s:|-]+\|\s*$", text, re.M))


def _count_cid(text: str) -> int:
    import re
    return len(re.findall(r"\(cid:\s*\d+\)", text))


def _headings(text: str) -> list[str]:
    import re
    return [m.group(2).strip()
            for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.M)]


# ---------------------------------------------------------------- 提取器


class MinerUExtractor:
    """MinerU 提取器。"""

    def __init__(self, config: ExtractConfig | None = None) -> None:
        self.cfg = config or ExtractConfig()

    # -------------------------------------------------- 可用性

    @staticmethod
    def available() -> bool:
        """检查 MinerU 是否可用。"""
        try:
            import importlib.util
            return importlib.util.find_spec("mineru") is not None
        except Exception:
            return False

    @staticmethod
    def cli_path() -> str | None:
        """查找 mineru 命令行。"""
        return shutil.which("mineru")

    # -------------------------------------------------- 库方式

    def _extract_via_api(self, pdf_path: Path) -> str:
        """通过 MinerU Python API 提取。"""
        from mineru.cli.common import do_parse  # type: ignore

        out_dir = Path(self.cfg.output_dir or
                       (pdf_path.parent / "_mineru_out"))
        out_dir.mkdir(parents=True, exist_ok=True)

        #构造 MinerU 配置：公式/表格识别开关在这里生效
        from mineru.cli.options import (  # type: ignore
            FormulaEnable, TableEnable,
        )
        cfg = {
            "formula_enable": FormulaEnable(
                self.cfg.formula_enable),
            "table_enable": TableEnable(
                self.cfg.table_enable),
        }

        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=[pdf_path.stem],
            pdf_bytes_list=[pdf_path.read_bytes()],
            p_lang_list=[self.cfg.lang],
            backend=self.cfg.backend,
            parse_method="auto",
            formula_enable=cfg["formula_enable"],
            table_enable=cfg["table_enable"],
        )

        return self._collect_markdown(out_dir)

    @staticmethod
    def _collect_markdown(out_dir: Path) -> str:
        """从 MinerU 输出目录收集 Markdown。

        MinerU 的产物结构：``<out>/<stem>/auto/<stem>.md``
        （不同版本层级略有差异，故做递归查找）
        """
        mds = sorted(out_dir.rglob("*.md"))
        if not mds:
            raise FileNotFoundError(f"MinerU 未产出 Markdown：{out_dir}")
        # 取最长的那个作为正文
        return max(mds, key=lambda p: p.stat().st_size).read_text(
            encoding="utf-8", errors="ignore")

    # -------------------------------------------------- 命令行方式

    def _extract_via_cli(self, pdf_path: Path) -> str:
        """通过命令行调用 MinerU。"""
        exe = self.cli_path()
        if not exe:
            raise FileNotFoundError("mineru 命令行不可用")

        out_dir = Path(self.cfg.output_dir or
                       (pdf_path.parent / "_mineru_out"))
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            exe, "-p", str(pdf_path), "-o", str(out_dir),
            "-b", self.cfg.backend, "-l", self.cfg.lang,
            "-d", self.cfg.device,
            "-m", "auto",
        ]
        # MinerU 2.x 早期版本用开关参数控制公式/表格
        if not self.cfg.formula_enable:
            cmd += ["-f", "false"]
        if not self.cfg.table_enable:
            cmd += ["-t", "false"]

        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mineru 退出码 {proc.returncode}: {proc.stderr[:400]}")

        return self._collect_markdown(out_dir)

    # -------------------------------------------------- 对外接口

    def extract(self, pdf_path: str | Path) -> ExtractResult:
        """提取单个 PDF。

        Returns
        -------
        ExtractResult
            失败时不抛异常，而是返回 ``ok=False`` 并带上原因，
            便于流水线批量处理时继续往下走。
        """
        import time

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return ExtractResult(ok=False, error=f"文件不存在: {pdf_path}")

        t0 = time.time()
        md = ""
        err = ""
        method = ""

        try:
            if self.available():
                try:
                    md = self._extract_via_api(pdf_path)
                    method = "api"
                except Exception as e1:
                    # API 失败回退命令行
                    if self.cli_path():
                        md = self._extract_via_cli(pdf_path)
                        method = "cli"
                    else:
                        raise RuntimeError(
                            f"API 调用失败（{type(e1).__name__}: {e1}），"
                            f"且无命令行可回退")
            elif self.cli_path():
                md = self._extract_via_cli(pdf_path)
                method = "cli"
            else:
                return ExtractResult(
                    ok=False,
                    error="MinerU 未安装。请执行 pip install -U 'mineru[core]'",
                )
        except Exception as exc:
            return ExtractResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_sec=time.time() - t0,
            )

        return ExtractResult(
            ok=True,
            markdown=md,
            n_chars=len(md),
            n_latex=_count_latex(md),
            n_tables=_count_tables(md),
            n_cid=_count_cid(md),
            headings=_headings(md),
            elapsed_sec=time.time() - t0,
            meta={
                "method": method,
                "config": self.cfg.to_dict(),
                "source": pdf_path.name,
            },
        )


def availability() -> dict:
    """检查 MinerU 环境，返回可读的状态报告。"""
    ex = MinerUExtractor()
    return {
        "api_available": MinerUExtractor.available(),
        "cli_available": bool(MinerUExtractor.cli_path()),
        "cli_path": MinerUExtractor.cli_path(),
        "usable": MinerUExtractor.available() or bool(MinerUExtractor.cli_path()),
    }
