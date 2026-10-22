"""
src/rag/agent.py — RAG Knowledge Retrieval Agent
Extractively assembles grounded, cited farmer advisory text from KB chapters.
Strictly extractive assembly only — NO LLM GENERATION [ENG Decision D19].
Chemical doses are hidden by default and returned only when include_doses=True [ENG Decision D23].
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.rag.retriever import ChapterResult, KBRetriever
from src.rag.ingest import KBChunk
from src.schemas.schemas import ChemicalOption, Citation, ClassProb, RAGOutput, VisionOutput

logger = logging.getLogger("leafmind.rag.agent")


def extract_citation_url_and_title(body: str) -> Tuple[Optional[str], str]:
    """
    Extract and reassemble citation URL that may wrap across lines in PDF.
    Guarantees no returned URL ends with a hyphen.
    """
    url_match = re.search(r"https?://[^\s]+", body)
    if not url_match:
        clean_title = re.sub(r"\s+", " ", body).strip()
        return None, clean_title

    url = url_match.group(0)
    url_start = url_match.start()
    after_url = body[url_match.end() :]

    # Reassemble wrapped fragments across line breaks
    while True:
        wrap_m = re.match(r"^\s*\n\s*([a-zA-Z0-9_\-\./\?=&%#]+)", after_url)
        if wrap_m:
            token = wrap_m.group(1).rstrip(".,;)")
            if url.endswith("-") or url.endswith("/"):
                url = url + token
                after_url = after_url[wrap_m.end() :]
                continue
            elif any(token.startswith(x) for x in ["fpls.", "f4cd-", "153-", "integrated-", "tomato-", "leaf-", "rots", "pepper-"]) or token.endswith((".html", ".pdf", ".php", "/")):
                url = url.rstrip("/") + "/" + token.lstrip("/")
                after_url = after_url[wrap_m.end() :]
                continue
        break

    url = url.rstrip(".,;)")
    if url.endswith("-"):
        url = url.rstrip("-")

    remaining = (body[:url_start] + " " + after_url).strip()
    clean_title = re.sub(r"\s+", " ", remaining).strip()
    return url, clean_title


def parse_citations_from_chunk(chunk: Optional[KBChunk]) -> List[Citation]:
    """Parse structured citations (tag, title, url) from Section 6 Sources chunk."""
    if not chunk or not chunk.text:
        return []

    citations: List[Citation] = []
    text = chunk.text

    # Match each [TAG] followed by text up to the next [TAG] or end of string
    pattern = r"\[([A-Za-z0-9_\-\.\s]{1,15})\]\s*([^\[]+)"
    for m in re.finditer(pattern, text):
        tag = m.group(1).strip()
        body = m.group(2).strip()
        url, title = extract_citation_url_and_title(body)
        if title:
            citations.append(Citation(tag=tag, title=title, url=url))

    return citations



def parse_chemical_options_from_chunk(
    chunk: Optional[KBChunk],
    include_doses: bool = False,
) -> List[ChemicalOption]:
    """
    Parse chemical options table from Section 5 Remedies chunk.
    Returns empty list unless include_doses=True [D23].
    """
    if not include_doses or not chunk or not chunk.text:
        return []

    options: List[ChemicalOption] = []
    lines = chunk.text.split("\n")

    merged_rows = []
    buffer = ""
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(line_clean.startswith(h) for h in ["Product", "Tag", "Verification tags", "Important cautions", "Caveats and regulatory"]):
            continue
        tag_match = re.search(r"\b(V[123]|US|--)\b", line_clean)
        if tag_match:
            full_row = (buffer + " " + line_clean).strip() if buffer else line_clean
            merged_rows.append(full_row)
            buffer = ""
        else:
            if not any(line_clean.startswith(h) for h in ["Biological", "Chemical options"]):
                buffer = (buffer + " " + line_clean).strip() if buffer else line_clean

    for row in merged_rows:
        tag_match = re.search(r"\b(V[123]|US|--)\b", row)
        if not tag_match:
            continue

        tag = tag_match.group(1)
        before_tag = row[: tag_match.start()].strip()
        after_tag = row[tag_match.end() :].strip()

        # Split product and dose/target
        parts = re.split(r"\s{2,}|\t", before_tag)
        if len(parts) >= 3:
            product = parts[0].strip()
            target = parts[1].strip()
            dose = parts[2].strip()
            water = parts[3].strip() if len(parts) > 3 else None
        else:
            product = before_tag
            dose = None
            water = None
            target = None

        options.append(
            ChemicalOption(
                product=product,
                verification_tag=tag,
                dose=dose,
                water=water,
                target=target,
                basis_phi=after_tag if after_tag else None,
            )
        )

    return options


def extract_what_it_is(
    glance_chunk: Optional[KBChunk],
    id_chunk: Optional[KBChunk],
) -> Optional[str]:
    """Extract disease definition and symptoms from glance and Section 1."""
    parts = []
    if glance_chunk and glance_chunk.text:
        parts.append(glance_chunk.text.strip())

    if id_chunk and id_chunk.text:
        id_text = id_chunk.text.strip()
        # Extract Symptoms section if present
        sym_match = re.search(r"(?i)Symptoms(.*?)(?=Distinguishing features|Look-alikes|Differential diagnosis|\Z)", id_text, re.DOTALL)
        if sym_match:
            sym_text = sym_match.group(1).strip()
            parts.append("Symptoms: " + sym_text)
        else:
            parts.append(id_text)

    return "\n\n".join(parts) if parts else None


def extract_why_it_happens(causes_chunk: Optional[KBChunk]) -> Optional[str]:
    """
    Extract causal organism and plain cause (survival, spread, progression) from Section 2.
    Filters out microscopic pathogen morphology descriptions, keeping pathogen name and plain causes.
    """
    if not causes_chunk or not causes_chunk.text:
        return None

    raw_sec2 = causes_chunk.text.strip()
    raw_bullets = re.split(r"(?:^|\n)\s*[\u2022\u25cf\-\*\x7f\ufffd]\s*", raw_sec2)
    cleaned_parts = []

    for b in raw_bullets:
        clean = re.sub(r"\s+", " ", b).strip()
        if not clean:
            continue
        # If bullet is pathogen identification line
        if re.search(r"\bPathogens?:\s*", clean, re.IGNORECASE):
            # Strip morphology (hyaline, non-septate, conidia, Gram, mycelium, rod-shaped, etc.)
            m_path = re.match(
                r"(Pathogens?:\s*[^;\-]+?)(?:\s*[-;]\s*(?:hyaline|non-septate|coloured|brownish|sporangia|conidia|Gram|mycelium|bi-flagellate|resting|thick-walled|acervuli|olive-grey|spindle|globose|three races|asexual|rod-shaped|motile|filamentous|flexuous|rigid|ssRNA|dsDNA).*|\s*\Z)",
                clean,
                re.IGNORECASE,
            )
            if m_path:
                p_text = m_path.group(1).strip()
                cites = re.findall(r"\([A-Za-z0-9_\-,\s]+\)", clean)
                cite_str = " " + cites[-1] if cites and not p_text.endswith(cites[-1]) else ""
                cleaned_parts.append(p_text.rstrip(".") + "." + cite_str)
            else:
                cleaned_parts.append(clean)
        else:
            cleaned_parts.append(clean)

    return "\n".join(cleaned_parts) if cleaned_parts else raw_sec2


def extract_what_to_do_first(
    prev_chunk: Optional[KBChunk],
    include_doses: bool = False,
) -> Optional[str]:
    """
    Build what_to_do_first from Section 4 Prevention and management bullets.
    Ordered so immediate actions on an infected crop (sanitation, removing infected parts,
    irrigation/drainage, a label-checked fungicide mention) come first and resistant-variety
    advice comes last.
    """
    if not prev_chunk or not prev_chunk.text:
        return None

    lines = [l.strip() for l in prev_chunk.text.split("\n") if l.strip()]
    if not lines:
        return None

    raw_bullets = re.split(r"(?:^|\n)\s*[\u2022\u25cf\-\*\x7f\ufffd]\s*", prev_chunk.text)
    bullets: List[str] = []
    for b in raw_bullets:
        clean = re.sub(r"\s+", " ", b).strip()
        if not clean:
            continue
        # Skip section headings or intros that aren't action bullets
        if any(clean.lower().startswith(h) for h in [
            "cultural practices", "prevention and management", "preventive measures",
            "sanitation and", "general measures"
        ]):
            continue
        if len(clean) > 15:
            bullets.append(clean)

    if not bullets:
        for l in lines:
            if not any(l.lower().startswith(h) for h in ["cultural", "prevention", "section"]):
                if len(l) > 15:
                    bullets.append(l)

    if not bullets:
        return lines[0]

    def score_bullet(bullet: str) -> int:
        """
        Lower score = higher priority (comes first).
        0: Immediate physical sanitation / removal of infected parts / hygiene / burning / destroying
        1: Immediate water / moisture / irrigation / drainage / airflow / staking / pruning
        2: Immediate label-checked spray / bioagent / fungicide / protectant mention
        3: Other cultural practices (crop rotation, planting date, weed control, spacing)
        4: Resistant-variety / hybrid / cultivar advice (comes LAST)
        """
        b_low = bullet.lower()
        is_resistant = any(k in b_low for k in [
            "resistant", "hybrid", "cultivar", "variety", "tolerant", "arka", "iihr",
            "ph-2", "ph-3", "resistance", "released hybrids", "non-host"
        ])
        is_sanitation = any(k in b_low for k in [
            "remove", "destroy", "burn", "bury", "dispose", "field hygiene", "clean",
            "infected plant parts", "infected parts", "infected leaves", "affected parts",
            "abnormal unharvested", "cut wood", "paste cut surfaces", "fallen leaves",
            "stubble", "sanitation", "debris", "cankered"
        ])
        is_drainage = any(k in b_low for k in [
            "drainage", "irrigation", "avoid overhead", "soil moisture", "mulch",
            "airflow", "spacing", "staking", "ventilate", "aeration"
        ])
        is_spray = any(k in b_low for k in [
            "spray", "fungicide", "bordeaux", "paste", "copper", "application",
            "trichoderma", "bacillus", "scout", "first spray", "protectant", "label"
        ])

        if is_sanitation:
            return 0
        if is_drainage:
            return 1
        if is_spray and not is_resistant:
            return 2
        if is_resistant:
            return 4
        return 3

    sorted_bullets = sorted(bullets, key=score_bullet)
    return sorted_bullets[0]



def extract_favourable_conditions(fav_chunk: Optional[KBChunk]) -> Optional[str]:
    """Extract weather and environmental drivers from Section 3."""
    if not fav_chunk or not fav_chunk.text:
        return None
    return fav_chunk.text.strip()


def extract_look_alikes(id_chunk: Optional[KBChunk]) -> Optional[str]:
    """Extract look-alikes and differential diagnosis from Section 1."""
    if not id_chunk or not id_chunk.text:
        return None

    text = id_chunk.text.strip()
    match = re.search(
        r"(?i)(?:Look-alikes|Distinguishing features|Differential diagnosis)(.*?)(?=\Z)",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(0).strip()
    return None


def extract_prevention(prev_chunk: Optional[KBChunk]) -> Optional[str]:
    """Extract full prevention and management measures from Section 4."""
    if not prev_chunk or not prev_chunk.text:
        return None
    return prev_chunk.text.strip()


class RAGAgent:
    """
    RAG Agent for knowledge retrieval.
    Processes VisionOutput (top-3 candidates) or direct candidate queries.
    Retrieval + extractive assembly only. No LLM generation.
    """

    def __init__(self, retriever: Optional[KBRetriever] = None):
        self.retriever = retriever or KBRetriever()

    def query_candidate(
        self,
        model_class: str,
        crop: Optional[str] = None,
        include_doses: bool = False,
    ) -> RAGOutput:
        """
        Retrieve and assemble grounded advisory for a single candidate model class.
        """
        retrieval: ChapterResult = self.retriever.retrieve_for_candidate(
            model_class=model_class,
            crop=crop,
        )

        if not retrieval.found:
            return RAGOutput(
                crop=retrieval.crop,
                model_class=model_class,
                disease_name=retrieval.disease_name,
                found=False,
                kb_gap=retrieval.kb_gap,
                reason=retrieval.reason,
                what_it_is=None,
                why_it_happens=None,
                what_to_do_first=None,
                favourable_conditions=None,
                look_alikes=None,
                prevention=None,
                chemical_options=[],
                caveats=[],
                citations=[],
                rag_available=not retrieval.kb_gap,
                knowledge_gap_flag=retrieval.kb_gap,
            )

        # Extractive assembly from chapter chunks
        what_it_is = extract_what_it_is(retrieval.glance_chunk, retrieval.identification_chunk)
        why_it_happens = extract_why_it_happens(retrieval.causes_chunk)
        what_to_do_first = extract_what_to_do_first(retrieval.prevention_chunk, include_doses=include_doses)
        favourable_conditions = extract_favourable_conditions(retrieval.favourable_chunk)
        look_alikes = extract_look_alikes(retrieval.identification_chunk)
        prevention = extract_prevention(retrieval.prevention_chunk)
        chemical_options = parse_chemical_options_from_chunk(
            retrieval.remedies_chunk,
            include_doses=include_doses,
        )
        caveats = list(retrieval.remedies_chunk.caveats) if retrieval.remedies_chunk else []
        citations = parse_citations_from_chunk(retrieval.sources_chunk)

        # Legacy fields synchronization
        treatment_summary = []
        if what_to_do_first:
            treatment_summary.append(what_to_do_first)
        if include_doses and chemical_options:
            treatment_summary.extend([f"{o.product} ({o.verification_tag})" for o in chemical_options[:3]])

        return RAGOutput(
            crop=retrieval.crop,
            model_class=model_class,
            disease_name=retrieval.disease_name,
            found=True,
            kb_gap=False,
            reason=None,
            what_it_is=what_it_is,
            why_it_happens=why_it_happens,
            what_to_do_first=what_to_do_first,
            favourable_conditions=favourable_conditions,
            look_alikes=look_alikes,
            prevention=prevention,
            chemical_options=chemical_options,
            caveats=caveats,
            citations=citations,
            rag_available=True,
            knowledge_gap_flag=False,
            treatment_options=treatment_summary,
            preventive_measures=[what_to_do_first] if what_to_do_first else [],
        )

    def process_vision_output(
        self,
        vision_output: VisionOutput,
        include_doses: bool = False,
    ) -> List[RAGOutput]:
        """
        Process VisionOutput: returns grounded RAGOutput per top-3 candidate.
        """
        crop = vision_output.crop
        candidates = vision_output.top_3_candidates or vision_output.top_k_probs

        results: List[RAGOutput] = []
        # Process up to top-3 candidates
        for candidate in candidates[:3]:
            label = candidate.label if hasattr(candidate, "label") else candidate.class_name
            ro = self.query_candidate(model_class=label, crop=crop, include_doses=include_doses)
            results.append(ro)

        return results
