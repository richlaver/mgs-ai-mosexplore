import logging
import re
import json
from typing import List, Optional
from classes import Execution, Context
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_classic import hub

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT_NAME = "langchain-ai/pairwise-evaluation-2"

def _strip_material_icons(text: str) -> str:
    """Remove material icon tokens of the form :material/<name>: from text."""
    if not text:
        return text
    return re.sub(r":material/[^:]+:", "", text)

def _build_candidate_text(ex: Execution) -> str:
    """Aggregate final response content plus artefact descriptions, stripping material icons."""
    parts: List[str] = []
    if ex.final_response and isinstance(ex.final_response, AIMessage):
        parts.append(_strip_material_icons(ex.final_response.content))
    for art in ex.artefacts:
        proc = art.additional_kwargs.get("process")
        if proc in ("plot", "csv"):
            parts.append(f"Artefact ({proc}): {_strip_material_icons(art.content)}")
    return "\n".join(parts).strip() or "(No content)"

def response_selector(llm: BaseLanguageModel, executions: List[Execution], context: Optional[Context]) -> List[Execution]:
    """Select best execution via progressive pairwise evaluation.

    If only one execution has a final response, mark it as best. If none have
    final responses, return executions unchanged. For >1, evaluate pairwise using
    hub prompt "langchain-ai/pairwise-evaluation-2". Tie-break: keep incumbent.
    """
    if not executions:
        return executions

    for ex in executions:
        if ex.is_best:
            ex.is_best = False

    candidates = [ex for ex in executions if ex.final_response is not None]
    if len(candidates) == 0:
        logger.info("response_selector: No candidates with final_response; skipping selection.")
        return executions
    if len(candidates) == 1:
        candidates[0].is_best = True
        return executions

    try:
        prompt = hub.pull(PAIRWISE_PROMPT_NAME)
    except Exception as e:
        logger.error(f"Failed to pull pairwise prompt: {e}; defaulting to first candidate.")
        candidates[0].is_best = True
        return executions

    incumbent = candidates[0]
    incumbent_text = _build_candidate_text(incumbent)

    for challenger in candidates[1:]:
        challenger_text = _build_candidate_text(challenger)
        chain = prompt | llm
        try:
            result = chain.invoke({
                "question": context.retrospective_query,
                "answer_a": incumbent_text,
                "answer_b": challenger_text,
            })
            output_text = getattr(result, "content", "") or str(result)
        except Exception as e:
            logger.error(f"Pairwise evaluation failed: {e}; keeping incumbent.")
            output_text = ""
        picked = None
        pref_num: Optional[int] = None
        if isinstance(result, dict):
            pref_raw = result.get("Preference") or result.get("preference") or result.get("preference_number")
            try:
                if pref_raw is not None:
                    pref_num = int(pref_raw)
            except Exception:
                pref_num = None
        if pref_num is None and hasattr(result, "additional_kwargs"):
            try:
                ak = getattr(result, "additional_kwargs", {}) or {}
                pref_raw = ak.get("Preference") or ak.get("preference")
                if pref_raw is not None:
                    pref_num = int(pref_raw)
            except Exception:
                pref_num = None
        if pref_num is None:
            try:
                parsed = json.loads(output_text)
                if isinstance(parsed, dict):
                    pref_raw = parsed.get("Preference") or parsed.get("preference")
                    if pref_raw is not None:
                        pref_num = int(pref_raw)
            except Exception:
                pref_num = None
        if pref_num is None:
            m = re.search(r"Preference\D*(1|2)", output_text, re.IGNORECASE)
            if m:
                try:
                    pref_num = int(m.group(1))
                except Exception:
                    pref_num = None
        if pref_num is None:
            text_lower = output_text.lower()
            if "answer a" in text_lower and "answer b" not in text_lower:
                picked = incumbent
            elif "answer b" in text_lower and "answer a" not in text_lower:
                picked = challenger
            else:
                picked = incumbent
        else:
            picked = incumbent if pref_num == 1 else challenger if pref_num == 2 else incumbent
        if picked is challenger:
            incumbent = challenger
            incumbent_text = challenger_text

    incumbent.is_best = True
    return executions
