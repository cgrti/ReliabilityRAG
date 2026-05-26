"""
Chunking + Metadata Pipeline for Turkish Integrated Reports (MinerU outputs)

Reads content_list.json from each report folder, groups content under headings,
splits into chunks with metadata (company, year, section_type, page, reliability_weight).
"""
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config import (
    DATA_ROOT,
    CHUNKS_OUTPUT,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    SECTION_KEYWORDS,
    SECTION_RELIABILITY_WEIGHTS,
)


@dataclass
class Chunk:
    chunk_id: str
    company: str
    year: int
    heading: str
    section_type: str
    reliability_weight: float
    page_start: int
    page_end: int
    text: str
    has_table: bool = False
    source_folder: str = ""


def parse_folder_name(folder_name: str) -> Optional[tuple[str, int]]:
    """Extract company name and year from folder like 'AdanaCimento_IR_2016'."""
    m = re.match(r"^(.+?)_IR_(\d{4})$", folder_name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def classify_section(text: str) -> str:
    """Classify a chunk's section type based on keyword matching."""
    text_lower = text.lower()
    scores = {}
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[section] = score
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "genel"
    return best


def html_table_to_text(html: str) -> str:
    """Convert HTML table to readable plain text."""
    # Remove tags but keep cell separators
    text = re.sub(r"<tr>", "", html)
    text = re.sub(r"</tr>", "\n", text)
    text = re.sub(r"<td[^>]*>", "", text)
    text = re.sub(r"</td>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    # Clean up
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip().rstrip("| ").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_report_items(folder_path: Path) -> list[dict]:
    """Load content_list.json from a report's auto/ directory."""
    auto_dir = folder_path / "auto"
    if not auto_dir.exists():
        return []
    json_files = [f for f in auto_dir.iterdir() if f.name.endswith("_content_list.json")]
    if not json_files:
        return []
    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def group_items_under_headings(items: list[dict]) -> list[dict]:
    """
    Group content items under their nearest preceding heading.
    Each group = {"heading": str, "page_start": int, "page_end": int, "texts": [str], "has_table": bool}
    """
    groups = []
    current = {
        "heading": "Genel",
        "page_start": 0,
        "page_end": 0,
        "texts": [],
        "has_table": False,
    }

    for item in items:
        page = item.get("page_idx", 0)

        if item["type"] == "text":
            text = item.get("text", "").strip()
            if not text:
                continue

            is_heading = item.get("text_level", -1) == 1

            if is_heading:
                # Save current group if it has content
                if current["texts"]:
                    groups.append(current)
                current = {
                    "heading": text,
                    "page_start": page,
                    "page_end": page,
                    "texts": [],
                    "has_table": False,
                }
            else:
                current["texts"].append(text)
                current["page_end"] = max(current["page_end"], page)

        elif item["type"] == "table":
            body = item.get("table_body", "")
            if body:
                table_text = html_table_to_text(body)
                caption = " ".join(item.get("table_caption", []))
                if caption:
                    table_text = f"[Tablo: {caption}]\n{table_text}"
                else:
                    table_text = f"[Tablo]\n{table_text}"
                current["texts"].append(table_text)
                current["has_table"] = True
                current["page_end"] = max(current["page_end"], page)

    # Don't forget last group
    if current["texts"]:
        groups.append(current)

    return groups


def split_group_into_chunks(
    group: dict,
    company: str,
    year: int,
    source_folder: str,
) -> list[Chunk]:
    """Split a heading group into sized chunks with overlap."""
    full_text = "\n".join(group["texts"])
    if len(full_text) < MIN_CHUNK_CHARS:
        return []

    section_type = classify_section(full_text)
    reliability = SECTION_RELIABILITY_WEIGHTS.get(section_type, 0.5)

    chunks = []

    if len(full_text) <= MAX_CHUNK_CHARS:
        # Single chunk — no split needed
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            company=company,
            year=year,
            heading=group["heading"][:200],
            section_type=section_type,
            reliability_weight=reliability,
            page_start=group["page_start"],
            page_end=group["page_end"],
            text=full_text,
            has_table=group["has_table"],
            source_folder=source_folder,
        ))
    else:
        # Split by paragraphs first, then by character limit
        paragraphs = full_text.split("\n")
        current_text = ""
        for para in paragraphs:
            if len(current_text) + len(para) + 1 > MAX_CHUNK_CHARS and current_text:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    company=company,
                    year=year,
                    heading=group["heading"][:200],
                    section_type=section_type,
                    reliability_weight=reliability,
                    page_start=group["page_start"],
                    page_end=group["page_end"],
                    text=current_text.strip(),
                    has_table=group["has_table"],
                    source_folder=source_folder,
                ))
                # Overlap: keep last CHUNK_OVERLAP_CHARS of previous chunk
                if CHUNK_OVERLAP_CHARS > 0 and len(current_text) > CHUNK_OVERLAP_CHARS:
                    current_text = current_text[-CHUNK_OVERLAP_CHARS:]
                else:
                    current_text = ""

            current_text += para + "\n"

        # Last piece
        if current_text.strip() and len(current_text.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                company=company,
                year=year,
                heading=group["heading"][:200],
                section_type=section_type,
                reliability_weight=reliability,
                page_start=group["page_start"],
                page_end=group["page_end"],
                text=current_text.strip(),
                has_table=group["has_table"],
                source_folder=source_folder,
            ))

    return chunks


def process_report(folder_path: Path) -> list[Chunk]:
    """Process a single report folder into chunks."""
    folder_name = folder_path.name
    parsed = parse_folder_name(folder_name)
    if not parsed:
        print(f"  [SKIP] Cannot parse folder: {folder_name}")
        return []

    company, year = parsed
    items = load_report_items(folder_path)
    if not items:
        print(f"  [SKIP] No content: {folder_name}")
        return []

    groups = group_items_under_headings(items)
    all_chunks = []
    for group in groups:
        chunks = split_group_into_chunks(group, company, year, folder_name)
        all_chunks.extend(chunks)

    return all_chunks


def process_all_reports() -> list[Chunk]:
    """Process all report folders under DATA_ROOT."""
    all_chunks = []
    folders = sorted([
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and "_IR_" in d.name
    ])

    print(f"Processing {len(folders)} report folders...")

    for i, folder in enumerate(folders):
        chunks = process_report(folder)
        all_chunks.extend(chunks)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(folders)}] Total chunks so far: {len(all_chunks)}")

    print(f"\nDone! Total chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: list[Chunk], output_path: Optional[Path] = None):
    """Save chunks to a JSONL file."""
    output_path = output_path or CHUNKS_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {output_path}")


def print_stats(chunks: list[Chunk]):
    """Print chunk statistics."""
    from collections import Counter

    print("\n=== CHUNK İSTATİSTİKLERİ ===")
    print(f"Toplam chunk: {len(chunks)}")

    # By section type
    section_counts = Counter(c.section_type for c in chunks)
    print("\nBölüm tipi dağılımı:")
    for st, cnt in section_counts.most_common():
        pct = cnt / len(chunks) * 100
        print(f"  {st}: {cnt} ({pct:.1f}%)")

    # By company count
    company_counts = Counter(c.company for c in chunks)
    print(f"\nŞirket sayısı: {len(company_counts)}")
    print("En çok chunk'a sahip 5 şirket:")
    for comp, cnt in company_counts.most_common(5):
        print(f"  {comp}: {cnt}")

    # Text length stats
    lengths = [len(c.text) for c in chunks]
    print(f"\nChunk uzunlukları:")
    print(f"  Ortalama: {sum(lengths)/len(lengths):.0f} karakter")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Tablo içeren chunk: {sum(1 for c in chunks if c.has_table)}")

    # Year distribution
    year_counts = Counter(c.year for c in chunks)
    print("\nYıl dağılımı:")
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]}")


if __name__ == "__main__":
    chunks = process_all_reports()
    print_stats(chunks)
    save_chunks(chunks)
