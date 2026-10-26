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


KNOWN_SUPPLEMENTARY_TAGS = {"TXP", "TXT", "TNT", "AUS", "AHDB", "UPL", "BRZ", "SHT"}

BROAD_ACTIVES = [
    "kasugamycin", "streptocycline", "streptomycin", "oxytetracycline",
    "mancozeb", "metalaxyl", "metalaxyl-m", "chlorothalonil",
    "copper oxychloride", "copper hydroxide", "copper sulphate", "bordeaux",
    "carbendazim", "azoxystrobin", "captan", "difenoconazole", "hexaconazole",
    "propiconazole", "tebuconazole", "trifloxystrobin", "cymoxanil",
    "dimethomorph", "fenamidone", "famoxadone", "thiophanate-methyl", "thiophanate",
    "myclobutanil", "sulphur", "sulfur", "urea spray", "urea sprays", "boric acid",
    "trichoderma", "pseudomonas",
]

RE_CHEMICAL_TARGETS = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(a) for a in BROAD_ACTIVES) + r")\b"
)

RE_DOSE_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:g|kg|mg|ml|l|L)\s*(?:/|per)\s*\d*\s*(?:kg|l|L|ha|acre|ac|litre|100\s*L)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:kg|g)/ac\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(?:SL|SP|EC|WP|WG|SC|DP|FS)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*%\s*\+\s*\d+(?:\.\d+)?\s*%\s*(?:SL|SP|EC|WP|WG|SC|DP|FS)?\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ppm|mg/L|g/L|ml/L)\b", re.I),
    re.compile(r"\b(?:at\s+)?(?:\d+(?:\.\d+)?\s*(?:g|kg|mg|ml|l|L)\s+)?per\s+100\s*L\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(?:boric(?:\s+acid)?|bordeaux(?:\s+(?:paste|mixture))?|urea(?:\s+spray)?)\b", re.I),
    re.compile(r"\b(?:boric(?:\s+acid)?|bordeaux(?:\s+(?:paste|mixture))?|urea(?:\s+spray)?)\s+\d+(?:\.\d+)?\s*%\b", re.I),
]


def scrub_doses_and_chemicals(text: Optional[str]) -> Optional[str]:
    """
    Remove application rates, concentrations, formulation strengths, and sentences
    recommending named pesticides, antibiotics or active ingredients when include_doses=False [D28].
    Preserves generic terms (copper, fungicide, bactericide) and resistance descriptions.
    """
    if not text:
        return text

    has_chemical = bool(RE_CHEMICAL_TARGETS.search(text))
    has_dose = any(bool(p.search(text)) for p in RE_DOSE_PATTERNS)
    if not has_chemical and not has_dose:
        return text

    # 1. Multi-line sentence / paragraph scrubs across line breaks
    cleaned_text = text

    # Chilli CIB&RC recommendation
    cleaned_text = re.sub(
        r"(?i)For chilli,\s*CIB&RC\s+lists\s+kasugamycin[^\.]*?bacterial leaf spot\.\s*",
        "",
        cleaned_text,
        flags=re.DOTALL,
    )
    cleaned_text = re.sub(
        r"(?i)\bplus\s+streptocycline(?:\s+sprays)?\s*;",
        ";",
        cleaned_text,
    )

    # Apple scab: urea spray bullet across line breaks (drop entire bullet)
    cleaned_text = re.sub(
        r"(?i)(?:^|\n)[\s\x7f•\-\*]*Because the sexual stage forms on fallen leaves.*?(?:the NHB text does not state its purpose\.|urea spray[^\n]*\.)[ \t]*",
        "\n",
        cleaned_text,
        flags=re.DOTALL,
    )

    # Grape esca: wound sealant with 5% boric acid paragraph (drop entire paragraph)
    cleaned_text = re.sub(
        r"(?i)(?:^|\n)[\s\x7f•\-\*]*A wound sealant of 5% boric acid.*?(?:These are US\s+products\.[ \t]*)",
        "\n",
        cleaned_text,
        flags=re.DOTALL,
    )

    # Grape esca: Bordeaux paste paragraph (drop entire paragraph)
    cleaned_text = re.sub(
        r"(?i)(?:^|\n)[\s\x7f•\-\*]*India:\s*chapter 10 gives Bordeaux paste.*?(?:tests\s+it against esca\.[ \t]*)",
        "\n",
        cleaned_text,
        flags=re.DOTALL,
    )

    lines = cleaned_text.split("\n")
    kept_lines = []
    for line in lines:
        l_str = line.strip()
        # Drop entire bullet if it purely recommends named chemical or rate
        if re.search(r"\b(?:trichoderma|pseudomonas)\b[^\n\.;]*?\b\d+(?:\.\d+)?\s*(?:g|kg)", l_str, re.I):
            continue
        if re.search(r"\b(?:urea\s+spray|5%\s*boric|bordeaux\s+paste)\b", l_str, re.I):
            continue

        # Drop any dangling fragment starting with "Because" that has no complete main action after the comma
        if re.match(r"^(?:[\x7f•\-\*]\s*)?Because\b", l_str, re.I):
            parts = l_str.split(",", 1)
            if len(parts) == 2:
                after_comma = parts[1].strip()
                has_action = bool(re.search(
                    r"\b(?:is|are|was|were|can|will|should|must|reduces|helps|controls|gives|prevents|requires|needs|spread|spreads|forms|formed|protect|protects|limit|limits)\b",
                    after_comma,
                    re.I
                ))
                ends_hanging = bool(re.search(r"\b(?:by|the|and|or|of|in|for|published by|schedule)\s*$", after_comma, re.I))
                if not has_action or ends_hanging:
                    continue
            else:
                continue

        # Drop orphaned lines left from dropped previous sentences
        if re.match(r"^(?:dieback and esca|it against esca|products\.)\b", l_str, re.I):
            continue

        cleaned_line = line
        # Scrub inline rate patterns
        for pat in RE_DOSE_PATTERNS:
            cleaned_line = pat.sub("", cleaned_line)

        cleaned_line = re.sub(r"\(\s*\)", "", cleaned_line)
        cleaned_line = re.sub(r"[ \t]+", " ", cleaned_line)
        cleaned_line = re.sub(r"\s+;", ";", cleaned_line)
        cleaned_line = re.sub(r"\s+\.", ".", cleaned_line)
        cleaned_line = re.sub(r"([.;])\s*([.;])", r"\1", cleaned_line)
        if cleaned_line.strip():
            kept_lines.append(cleaned_line)


    res = "\n".join(kept_lines)
    for pat in RE_DOSE_PATTERNS:
        res = pat.sub("", res)

    res = re.sub(r"\(\s*\)", "", res)
    res = re.sub(r"[ \t]+", " ", res)
    res = re.sub(r"\s+;", ";", res)
    res = re.sub(r"\s+\.", ".", res)
    res = re.sub(r"([.;])\s*([.;])", r"\1", res)
    return res.strip()


def is_tag_supplementary(tag: str, title: str, crop: Optional[str] = None) -> bool:
    """Determine if a citation tag represents a supplementary or secondary source."""
    t_upper = tag.upper().strip()
    if t_upper in KNOWN_SUPPLEMENTARY_TAGS:
        return True
    t_low = title.lower()
    if any(k in t_low for k in ["supplementary", "non-indian", "israel", "australia", "texas"]):
        return True
    if crop:
        crop_low = crop.lower()
        if crop_low in ["pepper", "pepper,_bell", "chilli"]:
            if "tomato" in t_low and "pepper" not in t_low and "chilli" not in t_low:
                return True
        elif crop_low == "tomato":
            if ("pepper" in t_low or "grape" in t_low) and "tomato" not in t_low:
                return True
    return False


def extract_what_to_do_first(
    prev_chunk: Optional[KBChunk],
    sources_chunk: Optional[KBChunk] = None,
    crop: Optional[str] = None,
    include_doses: bool = False,
) -> Optional[str]:
    """
    Build what_to_do_first from Section 4 Prevention and management bullets.
    Ordered so immediate actions on an infected crop (sanitation, removing infected parts,
    irrigation/drainage, clean seed) come first and resistant-variety advice comes last.
    Spray actions and rationales are rejected/penalized in favour of direct non-chemical imperatives.
    """
    if not prev_chunk or not prev_chunk.text:
        return None

    # Exclude Regional note / Maharashtra note / <Word> note section so it never attaches
    text = re.sub(r"(?i)(?:\n|^)\s*(?:Regional|Maharashtra|[A-Z][a-z]+)\s+note\b.*", "", prev_chunk.text, flags=re.DOTALL)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    raw_bullets = re.split(r"(?:^|\n)\s*[\u2022\u25cf\-\*\x7f\ufffd]\s*", text)
    citations = parse_citations_from_chunk(sources_chunk) if sources_chunk else []
    tag_map = {c.tag: c for c in citations}

    def tag_is_supp(tag: str) -> bool:
        c = tag_map.get(tag)
        title = c.title if c else ""
        return is_tag_supplementary(tag, title, crop)

    bullets: List[str] = []
    for b in raw_bullets:
        # Strip any trailing section note before continuation line joining
        b_clean_note = re.sub(r"(?i)(?:\n|^)\s*(?:Regional|Maharashtra|[A-Z][a-z]+)\s+note\b.*", "", b, flags=re.DOTALL)
        clean = re.sub(r"\s+", " ", b_clean_note).strip()
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
        b_low = bullet.lower()

        # Extract tags from bullet
        bullet_tags = []
        for m in re.finditer(r"\(([A-Za-z0-9_\-,\s]+)\)", bullet):
            for t in m.group(1).split(","):
                t_clean = t.strip()
                if t_clean in tag_map or t_clean in KNOWN_SUPPLEMENTARY_TAGS:
                    bullet_tags.append(t_clean)

        is_supp = False
        if bullet_tags:
            is_supp = all(tag_is_supp(t) for t in bullet_tags)
        elif any(k in b_low for k in ["supplementary", "non-indian", "other crop"]):
            is_supp = True

        supp_penalty = 30 if is_supp else 0

        # Rationale check: skip or penalise bullets starting with 'Because', 'In ...', or author descriptions
        is_rationale = bool(re.match(r"^(?:Because|In\s+[A-Za-z]+|Authors?|A\s+review|Studies)\b", bullet, re.I))
        rationale_penalty = 25 if is_rationale else 0

        # Urea spray / named chemical / rate penalty when include_doses=False
        has_chemical_rate = bool(
            re.search(r"\b(?:urea\s+spray|5%\s*boric|bordeaux\s+paste|kasugamycin|streptocycline)\b", b_low) or
            re.search(r"\b\d+(?:\.\d+)?\s*(?:g|kg|mg|ml|l|L)\s*(?:/|per)", b_low) or
            re.search(r"\bper\s+100\s*L\b", b_low)
        )
        chemical_rate_penalty = 50 if (not include_doses and has_chemical_rate) else 0

        # General pesticide spray penalty (starts with or advises spraying)
        has_pesticide_spray = bool(
            re.search(r"^(?:Time\s+sprays|Start\s+a\s+preventive\s+spray|Spray\s+programme)\b", bullet, re.I) or
            (not include_doses and re.search(r"\b(?:spray programme|sprays? started|pesticide spray)\b", b_low))
        )
        pesticide_penalty = 35 if has_pesticide_spray else 0

        # Descriptive resistance statement penalty (e.g. "Resistance to dodine... has been reported")
        is_descriptive_resistance = bool(re.match(r"^Resistance\s+to\b", bullet, re.I))
        resistance_stmt_penalty = 20 if is_descriptive_resistance else 0

        is_sanitation = any(k in b_low for k in [
            "remove", "destroy", "burn", "bury", "field hygiene", "clean",
            "infected plant parts", "infected parts", "infected leaves", "affected parts",
            "disease-free seed", "healthy transplants", "clean seed", "certified seed",
            "seed treatment", "stubble", "sanitation", "debris"
        ])
        is_drainage = any(k in b_low for k in [
            "drainage", "irrigation", "avoid overhead", "soil moisture", "mulch",
            "airflow", "spacing", "staking", "ventilate", "aeration"
        ])
        is_resistant = any(k in b_low for k in [
            "resistant", "hybrid", "cultivar", "variety", "tolerant", "arka", "iihr",
            "ph-2", "ph-3", "released hybrids"
        ])

        # Direct imperative check (including heading prefixes like "Resistance management: restrict...")
        is_imperative = bool(
            re.match(
                r"^(?:[A-Za-z\s]+:\s+)?(?:Use|Remove|Burn|Bury|Collect|Avoid|Plant|Grow|Raise|Reduce|Clean|Prune|Rotate|Ensure|Scout|Clear|Sanitize|Sanitise|Restrict|Mix|Alternate)\b",
                bullet,
                re.I,
            )
        ) and not has_pesticide_spray
        imperative_bonus = -3 if (is_imperative and not is_resistant) else 0

        base = 5
        if is_sanitation:
            base = 0
        elif is_drainage:
            base = 1
        elif is_resistant:
            base = 10  # Resistant variety comes LAST
        else:
            base = 2

        return base + supp_penalty + rationale_penalty + chemical_rate_penalty + pesticide_penalty + resistance_stmt_penalty + imperative_bonus

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
        what_to_do_first = extract_what_to_do_first(
            retrieval.prevention_chunk,
            sources_chunk=retrieval.sources_chunk,
            crop=retrieval.crop,
            include_doses=include_doses,
        )
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

        preventive_summary = [what_to_do_first] if what_to_do_first else []

        # Dose scrub at assembly time when include_doses=False [D28]
        if not include_doses:
            what_it_is = scrub_doses_and_chemicals(what_it_is)
            why_it_happens = scrub_doses_and_chemicals(why_it_happens)
            what_to_do_first = scrub_doses_and_chemicals(what_to_do_first)
            favourable_conditions = scrub_doses_and_chemicals(favourable_conditions)
            look_alikes = scrub_doses_and_chemicals(look_alikes)
            prevention = scrub_doses_and_chemicals(prevention)
            treatment_summary = [
                s for s in (scrub_doses_and_chemicals(t) for t in treatment_summary) if s
            ]
            preventive_summary = [
                s for s in (scrub_doses_and_chemicals(p) for p in preventive_summary) if s
            ]

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
            preventive_measures=preventive_summary,
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
