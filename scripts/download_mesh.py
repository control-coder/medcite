"""下载 MeSH descriptor 并解析为 synonym map。

MeSH (Medical Subject Headings) 是 NLM 的公开医学主题词表。
本脚本下载 MeSH ASCII 格式，解析为主题词 → 同义词列表的映射，
供阶段 4 术语归一化的第三层（外部标准映射）使用。

不宣称 UMLS 级能力，只是接入 MeSH 公开词表。

输出: src/medidiag/rag/medical_terms/mesh_synonyms.json
       {preferred_term: [synonym1, synonym2, ...]}

备选: 若 NLM 下载失败，可从 PubMedQA 的 MESHES 字段派生轻量词表。
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import click

# MeSH ASCII 下载地址（NLM 官方）
MESH_ASCII_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/2025/asciimesh/dmesh2025.bin"


def download_mesh(url: str, output_path: Path) -> None:
    """下载 MeSH 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"正在下载 MeSH: {url}")
    urllib.request.urlretrieve(url, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    click.echo(f"下载完成: {output_path} ({size_mb:.1f} MB)")


def parse_mesh_ascii(path: Path) -> dict[str, list[str]]:
    """解析 MeSH ASCII 格式，返回 {preferred_term: [synonyms]}。

    MeSH ASCII 格式:
      *NEWRECORD
      MH = Abdominal Neoplasms
      ENTRY = Abdominal Neoplasms|Neoplasms, Abdominal
      SY = Cancer of Abdomen
      ...
    """
    content = path.read_text(encoding="utf-8", errors="ignore")

    # 按 *NEWRECORD 分割
    records = content.split("*NEWRECORD")
    synonym_map: dict[str, list[str]] = {}

    for rec in records[1:]:  # 跳过第一个空块
        mh: str | None = None
        synonyms: list[str] = []

        for line in rec.split("\n"):
            line = line.strip()
            if line.startswith("MH = "):
                mh = line[5:].strip()
            elif line.startswith("ENTRY = "):
                entry = line[8:].strip()
                # ENTRY 格式: "Term|Variant1|Variant2" 或 "Term *VARIANT1*VARIANT2"
                for sep in ["|", "*"]:
                    if sep in entry:
                        entry = entry.replace(sep, "\n")
                for part in entry.split("\n"):
                    part = part.strip()
                    if part and part != mh:
                        synonyms.append(part)
            elif line.startswith("PRINT ENTRY = "):
                entry = line[14:].strip()
                for sep in ["|", "*"]:
                    if sep in entry:
                        entry = entry.replace(sep, "\n")
                for part in entry.split("\n"):
                    part = part.strip()
                    if part and part != mh:
                        synonyms.append(part)

        if mh:
            # 去重
            unique_syns = list(dict.fromkeys(synonyms))
            synonym_map[mh] = unique_syns

    return synonym_map


def derive_from_pubmedqa_meshes(
    pubmedqa_eval: str = "eval/datasets/eval_set_pubmedqa.jsonl",
) -> dict[str, list[str]]:
    """备选方案：从 PubMedQA 的 MESHES 字段派生轻量 synonym map。

    仅当 NLM 下载失败时使用。这不是完整 MeSH，只是数据集中出现过的主题词。
    """
    from medidiag.schemas import read_jsonl

    p = Path(pubmedqa_eval)
    if not p.exists():
        return {}

    records = read_jsonl(p)
    mesh_terms: set[str] = set()
    for rec in records:
        for m in rec.get("metadata", {}).get("meshes", []):
            mesh_terms.add(m)

    # 轻量 map：每个主题词映射到自身（无同义词，仅作为术语表）
    return {term: [] for term in mesh_terms}


@click.command()
@click.option(
    "--output-dir", default="eval/datasets/raw/mesh",
    show_default=True,
    help="MeSH 原始文件存放目录。",
)
@click.option(
    "--synonym-output", default="src/medidiag/rag/medical_terms/mesh_synonyms.json",
    show_default=True,
    help="解析后的 synonym map 输出路径。",
)
@click.option(
    "--fallback-pubmedqa", is_flag=True, default=False,
    help="跳过 NLM 下载，从 PubMedQA MESHES 字段派生轻量词表。",
)
def cli(output_dir: str, synonym_output: str, fallback_pubmedqa: bool) -> None:
    """下载 MeSH descriptor 并解析为 synonym map。"""

    out_path = Path(synonym_output)

    if fallback_pubmedqa:
        click.echo("使用备选方案：从 PubMedQA MESHES 派生轻量词表")
        synonym_map = derive_from_pubmedqa_meshes()
    else:
        mesh_file = Path(output_dir) / "dmesh2025.bin"

        if not mesh_file.exists():
            try:
                download_mesh(MESH_ASCII_URL, mesh_file)
            except Exception as e:
                click.echo(f"下载失败: {e}")
                click.echo("切换到备选方案：从 PubMedQA MESHES 派生")
                synonym_map = derive_from_pubmedqa_meshes()
                _write_and_report(synonym_map, out_path, source="PubMedQA_MESHES_fallback")
                return
        else:
            click.echo(f"MeSH 文件已存在: {mesh_file}")

        click.echo("解析 MeSH ASCII...")
        synonym_map = parse_mesh_ascii(mesh_file)

    _write_and_report(synonym_map, out_path, source="MeSH_descriptor" if not fallback_pubmedqa else "PubMedQA_MESHES_fallback")


def _write_and_report(synonym_map: dict[str, list[str]], out_path: Path, source: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(synonym_map, f, ensure_ascii=False, indent=2)

    total_terms = len(synonym_map)
    total_synonyms = sum(len(v) for v in synonym_map.values())
    terms_with_syn = sum(1 for v in synonym_map.values() if v)

    click.echo("")
    click.echo("========== MeSH Synonym Map ==========")
    click.echo(f"来源            : {source}")
    click.echo(f"主题词数        : {total_terms}")
    click.echo(f"有同义词的主题词: {terms_with_syn}")
    click.echo(f"同义词总数      : {total_synonyms}")
    click.echo(f"输出            : {out_path}")


if __name__ == "__main__":
    cli()
