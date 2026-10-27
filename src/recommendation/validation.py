"""
src/recommendation/validation.py
================================
Grounding and output validation for Recommendation Agent (Stage C6).

Ensures that the LLM has only reworded and organised provided input:
  - Validates valid JSON structure.
  - Verifies that no numbers appear in output that were absent from the input.
  - Verifies that no chemical products appear in output that were absent from the input.
  - Verifies that no application doses or rates appear when include_doses=False.
  - Rejects empty or malformed outputs.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.rag.agent import BROAD_ACTIVES, RE_DOSE_PATTERNS

# Regex for numbers (integers, decimals, percentages, e.g. 5, 2.5, 90%, 3-5)
RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")

# Standard numbers allowed in advisory phrasing (intervals, hours, steps)
STANDARD_ALLOWED_NUMBERS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "24", "48", "72", "100", "0", "0.0", "1.0",
}


def extract_numbers(text: str) -> Set[str]:
    """Extract normalized set of numeric tokens from text."""
    if not text:
        return set()
    matches = RE_NUMBER.findall(text)
    return {m.lower().rstrip("%") for m in matches}


def extract_products(text: str) -> Set[str]:
    """Extract chemical actives and product names found in text."""
    if not text:
        return set()
    found = set()
    text_lower = text.lower()
    for active in BROAD_ACTIVES:
        # Match whole word
        pattern = r"\b" + re.escape(active.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(active.lower())
    return found


def clean_step_indices(steps: List[str]) -> List[str]:
    """Strip leading step numbers (e.g. '1. ', '(2) ') from steps list."""
    cleaned = []
    for s in steps:
        c = re.sub(r"^\s*(?:\d+[\.\)]|\(?\d+\))\s*", "", str(s))
        cleaned.append(c)
    return cleaned


def validate_grounding(
    raw_response: str,
    input_prompt: str,
    include_doses: bool = False,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate that raw LLM response adheres strictly to grounding constraints.

    Args:
        raw_response: The raw string response from LLM provider.
        input_prompt: The prompt text that was sent to the LLM.
        include_doses: Whether chemical doses were permitted.

    Returns:
        (is_valid, failure_reason, parsed_data_dict)
    """
    if not raw_response or not raw_response.strip():
        return False, "empty_llm_output", None

    clean_text = raw_response.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
    except Exception as e:
        return False, f"json_decode_error: {str(e)}", None

    if not isinstance(data, dict):
        return False, "llm_output_not_a_json_object", None

    summary = str(data.get("summary", "") or "")
    what_to_do_first = str(data.get("what_to_do_first", "") or "")
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        return False, "steps_field_not_a_list", None
    steps = [str(s) for s in raw_steps if s is not None]
    weather_note = str(data.get("weather_note", "") or "")
    uncertainty_note = str(data.get("uncertainty_note", "") or "")

    if not summary and not what_to_do_first and not steps:
        return False, "empty_llm_output_fields", None

    # Clean leading list numbers from steps for number validation
    cleaned_steps = clean_step_indices(steps)
    combined_output = " ".join([summary, what_to_do_first] + cleaned_steps + [weather_note, uncertainty_note])

    # 1. Check for doses if include_doses=False (D23/D28)
    if not include_doses:
        for pattern in RE_DOSE_PATTERNS:
            m = pattern.search(combined_output)
            if m:
                return False, f"dose_pattern_detected: '{m.group(0)}'", None

    # 2. Check for ungrounded chemical products
    output_products = extract_products(combined_output)
    input_products = extract_products(input_prompt)

    ungrounded_products = output_products - input_products
    if ungrounded_products:
        bad_prod = next(iter(ungrounded_products))
        return False, f"ungrounded_product: '{bad_prod}' not in input", None

    # 3. Check for ungrounded numbers
    output_numbers = extract_numbers(combined_output)
    input_numbers = extract_numbers(input_prompt).union(STANDARD_ALLOWED_NUMBERS)

    ungrounded_numbers = output_numbers - input_numbers
    if ungrounded_numbers:
        bad_num = next(iter(ungrounded_numbers))
        return False, f"ungrounded_number: '{bad_num}' not in input", None

    parsed_result = {
        "summary": summary,
        "what_to_do_first": what_to_do_first if what_to_do_first else None,
        "steps": steps,
        "weather_note": weather_note if weather_note else None,
        "uncertainty_note": uncertainty_note if uncertainty_note else None,
    }

    return True, None, parsed_result
