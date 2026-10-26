"""
src/rag/ingest.py — Knowledge Base Ingestion Pipeline
Parses disease PDF manuals in data/knowledge_base/, chunks by chapter and section,
extracts verification tags and source citations, attaches statutory caveats, and
maps diseases to model classes using config/kb_class_map.yaml.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import yaml
from pypdf import PdfReader

logger = logging.getLogger("leafmind.rag.ingest")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = REPO_ROOT / "data" / "knowledge_base"
DEFAULT_CLASS_MAP_PATH = REPO_ROOT / "config" / "kb_class_map.yaml"

CROP_CAVEATS: Dict[str, List[str]] = {
    "Tomato": [
        "Always check the current label and the live register (cibrc.gov.in) and follow the pre-harvest interval (PHI, in days) printed on the label.",
        "Fungicide entries tagged V1 come from the CIB&RC compilation up to 30/09/2025; insecticide entries come from older CIB&RC versions (2019/2021). Registrations, label claims and permitted uses change; some listed active ingredients may since have been restricted or withdrawn."
    ],
    "Apple": [
        "Always check the current label and the live register (cibrc.gov.in) and follow the pre-harvest interval (PHI, in days) printed on the label.",
        "The CIB&RC compilation used for V1 is dated up to 30/09/2025 and was read from a copy hosted on a university website, not from the live register. Registrations, label claims and permitted uses change."
    ],
    "Cherry": [
        "Always check the current label and the live register (cibrc.gov.in) and follow the pre-harvest interval (PHI, in days) printed on the label.",
        "CIB&RC Major Uses of Pesticides - Fungicides compilation up to 30/09/2025. Registrations, label claims and permitted uses change."
    ],
    "Grape": [
        "Always check the current label and the live register before use and follow the pre-harvest interval (PHI, in days) printed on the label.",
        "ICAR-NRC Grapes label-claim list A5 (revised 29 October 2024); CIB&RC waiting-period data compiled from Major Uses of Pesticides (01/01/2021). Check residue limits before use in export vineyards."
    ],
    "Corn": [
        "Always check the current label and the live CIB&RC register (ppqs.gov.in) and follow the pre-harvest interval (PHI, in days) printed on the label.",
        "The CIB&RC compilation used for V1 is 'based on certificates issued' up to 30.09.2025. Registrations and labels change."
    ],
    "Blueberry": [
        "No fungicide or bactericide claim for blueberry was matched to the CIB&RC compilation, and no Indian university package for blueberry was found. Confirm current legal status before use.",
        "Chemical information is shown only as active ingredients with US efficacy ratings (tag US), without doses, for reference."
    ],
    "Potato": [
        "Always check the current label and the live CIB&RC register (ppqs.gov.in / cibrc.gov.in) and follow the pre-harvest interval (PHI) printed on the label.",
        "The CIB&RC compilation used for V1 is 'based on certificate issued' up to 30.09.2025. Registrations and labels change; verify product registration and permitted uses before application."
    ],
    "Pepper,_bell": [
        "Always check the current label and the live CIB&RC register (ppqs.gov.in / cibrc.gov.in) and follow the pre-harvest interval (PHI) printed on the label.",
        "The CIB&RC compilation used for V1 is dated up to 30.09.2025 where rows name the crop 'chilli'; confirm product registration and label before use on capsicum / bell pepper, especially under protected cultivation."
    ]
}

DEFAULT_CAVEATS: List[str] = [
    "Always check the current label and the live CIB&RC register (cibrc.gov.in) and follow the pre-harvest interval (PHI, in days) printed on the label.",
    "Verify product registration, permitted uses, and dosage on the latest CIB&RC compilation before application."
]

STANDARD_SECTIONS = [
    "Identification",
    "Causes and progression",
    "Favourable conditions",
    "Prevention and management",
    "Remedies and treatment",
    "Sources and citations",
]


@dataclass
class KBChunk:
    chunk_id: str
    crop: str
    disease_name: str
    model_class: Optional[str]
    section: str
    chunk_type: str  # "glance" | "section" | "sources" | "intro" | "summary" | "appendix"
    verification_tags: List[str] = field(default_factory=list)
    source_tags: List[str] = field(default_factory=list)
    source_file: str = ""
    page: int = 1
    text: str = ""
    caveats: List[str] = field(default_factory=list)
    indexed_for_farmer: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KBChunk:
        return cls(**d)


def infer_crop_from_filename(filename: str, kb_class_map: Dict[str, Any]) -> str:
    """Infer canonical crop name from PDF filename using kb_class_map aliases."""
    name_lower = filename.lower()
    for crop, data in kb_class_map.get("crops", {}).items():
        if crop.lower() in name_lower:
            return crop
        for alias in data.get("aliases", []):
            if alias.lower() in name_lower:
                return crop
    # Fallback: first word
    first = filename.split("_")[0].split(" ")[0].capitalize()
    return first


def map_disease_to_model_class(
    crop: str, disease_title: str, kb_class_map: Dict[str, Any]
) -> Optional[str]:
    """Map a disease title to its exact model class from kb_class_map."""
    crop_data = kb_class_map.get("crops", {}).get(crop)
    if not crop_data:
        return None
    diseases = crop_data.get("diseases", {})
    if disease_title in diseases:
        return diseases[disease_title].get("model_class")

    clean_title = disease_title.strip().lower()
    for d_name, d_info in diseases.items():
        if d_name.lower() == clean_title:
            return d_info.get("model_class")
        for alias in d_info.get("aliases", []):
            a_clean = alias.lower()
            if a_clean == clean_title or a_clean in clean_title or clean_title in a_clean:
                return d_info.get("model_class")
    return None


# ---------------------------------------------------------------------------
# Running-header / page-footer patterns that every PDF embeds in the text
# layer. These must be stripped per page BEFORE building full_text so that
# they never bleed into section content.
# Pattern 1 (most crops):  "  Crop diseases - master document | Page N"
# Pattern 2 (Grape):       "  Grape diseases - combined reference | Compiled … | Page N"
# Pattern 3 (Cherry):      "Cherry diseases - combined reference          Page N"
# ---------------------------------------------------------------------------
_PAGE_FOOTER_RE = re.compile(
    r"^[ \t]*[^\n]*\bdiseases\b[^\n]*\bPage\s+\d+\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Ensure "Regional note" always begins on its own line (never mid-sentence).
_REGIONAL_NOTE_RE = re.compile(
    r"(?<![\n])(?=Regional note)",
    re.IGNORECASE,
)


def strip_page_running_header_footer(page_text: str) -> str:
    """Remove running header/footer lines and normalise 'Regional note' breaks."""
    # Remove any line matching the footer pattern
    cleaned = _PAGE_FOOTER_RE.sub("", page_text)
    # Ensure 'Regional note' starts on a new line (handles page-boundary joins)
    cleaned = _REGIONAL_NOTE_RE.sub("\n", cleaned)
    return cleaned


def extract_verification_tags(text: str) -> List[str]:
    """Extract verification tags V1, V2, V3, US, -- from text."""
    tags: Set[str] = set()
    for m in re.finditer(r"\b(V[123]|US|--)\b", text):
        t = m.group(1)
        tags.add(t)
    return sorted(tags)


def parse_pdf_document(
    pdf_path: Path,
    kb_class_map: Dict[str, Any],
) -> Tuple[List[KBChunk], Dict[str, Any]]:
    """
    Parse a single PDF manual into structured chunks.
    Extracts:
      - Intro / About this document
      - Summary table
      - Disease chapters (glance + 6 standard sections)
      - Consolidated sources / appendices
    Reports any chapter with missing sections.
    """
    filename = pdf_path.name
    crop = infer_crop_from_filename(filename, kb_class_map)
    caveats = CROP_CAVEATS.get(crop, DEFAULT_CAVEATS)

    reader = PdfReader(str(pdf_path))
    pages_text: List[Tuple[int, str]] = []
    full_text = ""
    page_offsets: List[Tuple[int, int]] = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text(extraction_mode="layout") or ""
        txt = strip_page_running_header_footer(raw)
        page_offsets.append((len(full_text), i + 1))
        full_text += txt + "\n"
        pages_text.append((i + 1, txt))

    def get_page_number(char_offset: int) -> int:
        cur_p = 1
        for off, p in page_offsets:
            if char_offset >= off:
                cur_p = p
            else:
                break
        return cur_p

    chunks: List[KBChunk] = []
    stats: Dict[str, Any] = {
        "crop": crop,
        "filename": filename,
        "total_pages": len(pages_text),
        "chapters_found": 0,
        "missing_sections": {},
    }

    std_names = {
        "Identification",
        "Causes and progression",
        "Favourable conditions",
        "Prevention and management",
        "Remedies and treatment",
        "Sources and citations",
    }

    lines = full_text.split("\n")
    line_offsets: List[int] = []
    cur_offset = 0
    for l in lines:
        line_offsets.append(cur_offset)
        cur_offset += len(l) + 1

    # 1. Locate Chapter 1 start (skipping TOC and introductory summary table)
    ch1_start = 0
    ch1_match_idx = None
    for i, line in enumerate(lines):
        line_clean = line.strip()
        m = re.match(r"^1\.\s+([A-Za-z].*)$", line_clean)
        if m and m.group(1).strip() not in std_names:
            window = "\n".join(lines[i : min(len(lines), i + 40)])
            if "At a glance" in window and re.search(r"^\s*1\.?\s*Identification", window, re.MULTILINE):
                ch1_start = line_offsets[i]
                ch1_match_idx = i
                break

    if ch1_match_idx is None:
        logger.error("Could not find start of Chapter 1 in %s", filename)
        return chunks, stats

    # 2. Extract Intro and Summary table (before Ch 1)
    summary_match = re.search(r"^\s*Summary table", full_text[:ch1_start], re.MULTILINE)
    if summary_match:
        intro_text = full_text[: summary_match.start()].strip()
        summary_text = full_text[summary_match.start() : ch1_start].strip()
        summary_pos = summary_match.start()
    else:
        about_match = re.search(r"(?i)About this document", full_text[:ch1_start])
        if about_match:
            intro_text = full_text[about_match.start() : ch1_start].strip()
        else:
            intro_text = full_text[:ch1_start].strip()
        summary_text = ""
        summary_pos = ch1_start

    if intro_text:
        chunks.append(
            KBChunk(
                chunk_id=f"{crop}__intro",
                crop=crop,
                disease_name="Document Overview",
                model_class=None,
                section="About this document",
                chunk_type="intro",
                verification_tags=extract_verification_tags(intro_text),
                source_tags=[],
                source_file=filename,
                page=1,
                text=intro_text,
                caveats=caveats,
                indexed_for_farmer=False,
            )
        )

    if summary_text:
        summary_page = get_page_number(summary_pos)
        chunks.append(
            KBChunk(
                chunk_id=f"{crop}__summary_table",
                crop=crop,
                disease_name="Summary Table",
                model_class=None,
                section="Summary table",
                chunk_type="summary",
                verification_tags=extract_verification_tags(summary_text),
                source_tags=[],
                source_file=filename,
                page=summary_page,
                text=summary_text,
                caveats=caveats,
                indexed_for_farmer=False,
            )
        )

    # 4. Find all chapter boundaries
    chapter_headers: List[Tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        if line_offsets[i] < ch1_start:
            continue
        line_clean = line.strip()
        m = re.match(r"^(\d{1,2})\.\s+([A-Za-z].*)$", line_clean)
        if m:
            num = int(m.group(1))
            t1 = m.group(2).strip()
            if t1 in std_names:
                continue

            ahead = [lines[j].strip() for j in range(i + 1, min(len(lines), i + 8)) if lines[j].strip()]
            crop_kw = {"apple", "cherry", "grape", "tomato", "maize", "blueberry", "corn", "prunus", "vitis", "potato", "pepper", "chilli", "capsicum", "solanum"}
            for c_name, c_info in kb_class_map.get("crops", {}).items():
                crop_kw.add(c_name.lower())
                for a in c_info.get("aliases", []):
                    crop_kw.add(a.lower())

            pipe_idx = None
            for idx, al in enumerate(ahead):
                if "|" in al and ("crop" in al.lower() or "fungal" in al.lower() or "bacterial" in al.lower() or "oomycete" in al.lower() or "viral" in al.lower() or any(c in al.lower() for c in crop_kw)):
                    pipe_idx = idx
                    break

            if pipe_idx is not None:
                title_parts = [t1]
                for k in range(pipe_idx):
                    title_parts.append(ahead[k])
                title = re.sub(r"\s+", " ", " ".join(title_parts)).strip()
                title = re.sub(r"\s+/\s+", " / ", title)
                window = "\n".join(lines[i : min(len(lines), i + 40)])
                if "At a glance" in window:
                    chapter_headers.append((num, title, line_offsets[i]))

    stats["chapters_found"] = len(chapter_headers)

    # Find where disease chapters end (Consolidated source list, Appendix, or end of text)
    end_marker = re.search(r"^\s*(Consolidated source list|Appendix\s+[A-Z])", full_text[ch1_start:], re.MULTILINE)
    chapters_end_pos = ch1_start + end_marker.start() if end_marker else len(full_text)

    # 5. Process each chapter
    for idx, (ch_num, ch_title, ch_start) in enumerate(chapter_headers):
        ch_end = chapter_headers[idx + 1][2] if idx + 1 < len(chapter_headers) else chapters_end_pos
        ch_text = full_text[ch_start:ch_end]
        ch_page = get_page_number(ch_start)
        model_class = map_disease_to_model_class(crop, ch_title, kb_class_map)

        # First, find Section 6 to extract all explicitly defined citation tags in this chapter
        sec6_match = re.search(r"^\s*6\.?\s*Sources and citations(.*?)(?=\Z)", ch_text, re.MULTILINE | re.DOTALL)
        defined_sources: Set[str] = set()
        if sec6_match:
            for stag in re.finditer(r"\[([A-Za-z0-9_\-\.]{1,12})\]", sec6_match.group(1)):
                defined_sources.add(stag.group(1).strip())

        def find_cited_sources(text_segment: str) -> List[str]:
            cited: Set[str] = set()
            for stag in re.finditer(r"\[([A-Za-z0-9_\-\.]{1,12})\]", text_segment):
                t = stag.group(1).strip()
                if not defined_sources or t in defined_sources:
                    cited.add(t)
            # Also match parenthesized citations like (T8, IH26)
            for m in re.finditer(r"\(([A-Za-z0-9_\-]{1,10}(?:\s*,\s*[A-Za-z0-9_\-]{1,10})*)\)", text_segment):
                for part in m.group(1).split(","):
                    p = part.strip()
                    if p in defined_sources:
                        cited.add(p)
            return sorted(cited)

        missing_in_ch = []

        # (a) "At a glance" chunk
        glance_m = re.search(r"At a glance\.?\s*(.*?)(?=^\s*1\.?\s*Identification|\Z)", ch_text, re.MULTILINE | re.DOTALL)
        if glance_m and glance_m.group(1).strip():
            g_text = glance_m.group(1).strip()
            g_page = get_page_number(ch_start + glance_m.start())
            chunks.append(
                KBChunk(
                    chunk_id=f"{crop}__{ch_num}_glance",
                    crop=crop,
                    disease_name=ch_title,
                    model_class=model_class,
                    section="At a glance",
                    chunk_type="glance",
                    verification_tags=extract_verification_tags(g_text),
                    source_tags=find_cited_sources(g_text),
                    source_file=filename,
                    page=g_page,
                    text=g_text,
                    caveats=caveats,
                    indexed_for_farmer=True,
                )
            )
        else:
            missing_in_ch.append("At a glance")

        # (b) Sections 1 to 6
        section_regexes = [
            (1, "Identification", r"^\s*1\.?\s*Identification(.*?)(?=^\s*2\.?\s*Causes and progression|\Z)"),
            (2, "Causes and progression", r"^\s*2\.?\s*Causes and progression(.*?)(?=^\s*3\.?\s*Favourable conditions|\Z)"),
            (3, "Favourable conditions", r"^\s*3\.?\s*Favourable conditions(.*?)(?=^\s*4\.?\s*Prevention and management|\Z)"),
            (4, "Prevention and management", r"^\s*4\.?\s*Prevention and management(.*?)(?=^\s*5\.?\s*Remedies and treatment|\Z)"),
            (5, "Remedies and treatment", r"^\s*5\.?\s*Remedies and treatment(.*?)(?=^\s*6\.?\s*Sources and citations|\Z)"),
            (6, "Sources and citations", r"^\s*6\.?\s*Sources and citations(.*?)(?=\Z)"),
        ]

        for sec_num, sec_name, pattern in section_regexes:
            sm = re.search(pattern, ch_text, re.MULTILINE | re.DOTALL)
            if not sm or not sm.group(1).strip():
                missing_in_ch.append(f"{sec_num}. {sec_name}")
                continue

            sec_content = sm.group(1).strip()
            sec_page = get_page_number(ch_start + sm.start())
            chunk_type = "sources" if sec_num == 6 else "section"

            # Attach label caveat and CIB&RC date note to every Remedies chunk
            if sec_num == 5:
                caveat_suffix = "\n\nCaveats and regulatory notes:\n- " + "\n- ".join(caveats)
                sec_content += caveat_suffix

            chunks.append(
                KBChunk(
                    chunk_id=f"{crop}__{ch_num}_sec{sec_num}",
                    crop=crop,
                    disease_name=ch_title,
                    model_class=model_class,
                    section=sec_name,
                    chunk_type=chunk_type,
                    verification_tags=extract_verification_tags(sec_content),
                    source_tags=find_cited_sources(sec_content) if sec_num != 6 else sorted(defined_sources),
                    source_file=filename,
                    page=sec_page,
                    text=sec_content,
                    caveats=caveats if sec_num == 5 else [],
                    indexed_for_farmer=True,
                )
            )

        if missing_in_ch:
            stats["missing_sections"][f"Ch {ch_num} ({ch_title})"] = missing_in_ch
            logger.warning(
                "Chapter %d (%s) in %s has missing sections: %s",
                ch_num,
                ch_title,
                filename,
                missing_in_ch,
            )

    # 6. Extract Appendix / Consolidated source list if present (stored, not indexed for farmer answers)
    if end_marker:
        tail_text = full_text[chapters_end_pos:].strip()
        tail_page = get_page_number(chapters_end_pos)
        is_appendix = "Appendix" in tail_text[:100]
        chunks.append(
            KBChunk(
                chunk_id=f"{crop}__appendix",
                crop=crop,
                disease_name="Appendix / Consolidated Sources",
                model_class=None,
                section="Appendix" if is_appendix else "Consolidated source list",
                chunk_type="appendix" if is_appendix else "sources",
                verification_tags=extract_verification_tags(tail_text),
                source_tags=[],
                source_file=filename,
                page=tail_page,
                text=tail_text,
                caveats=caveats,
                indexed_for_farmer=False,
            )
        )

    return chunks, stats


def ingest_knowledge_base(
    kb_dir: Optional[Path] = None,
    class_map_path: Optional[Path] = None,
) -> Tuple[List[KBChunk], List[Dict[str, Any]]]:
    """
    Ingest all PDF manuals from kb_dir.
    Returns:
      chunks: All parsed KBChunk objects.
      reports: Summary report list per PDF.
    """
    kb_dir = kb_dir or DEFAULT_KB_DIR
    class_map_path = class_map_path or DEFAULT_CLASS_MAP_PATH

    with open(class_map_path, "r", encoding="utf-8") as f:
        kb_class_map = yaml.safe_load(f)

    all_chunks: List[KBChunk] = []
    reports: List[Dict[str, Any]] = []

    pdf_files = sorted(list(kb_dir.glob("*.pdf")))
    logger.info("Found %d PDF file(s) in %s", len(pdf_files), kb_dir)

    for pdf_file in pdf_files:
        chunks, report = parse_pdf_document(pdf_file, kb_class_map)
        all_chunks.extend(chunks)
        reports.append(report)

    return all_chunks, reports
