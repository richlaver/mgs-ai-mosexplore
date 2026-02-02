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
        artefact_type = str(art.get("type") or "")
        if artefact_type in ("plot", "csv"):
            description = str(art.get("description") or "")
            tool_name = str(art.get("tool_name") or "")
            artefact_id = str(art.get("id") or "")
            details = ", ".join(
                [
                    item
                    for item in [
                        f"id={artefact_id}" if artefact_id else "",
                        f"tool={tool_name}" if tool_name else "",
                        description,
                    ]
                    if item
                ]
            )
            parts.append(f"Artefact ({artefact_type}): {_strip_material_icons(details)}")
    return "\n".join(parts).strip() or "(No content)"

def response_selector(
    llm: BaseLanguageModel,
    executions: List[Execution],
    context: Optional[Context],
    response_mode: str | None = None,
    successful_executions: Optional[List[dict]] = None,
    response_facts: Optional[List[dict]] = None,
) -> List[Execution]:
    """Select best execution.

    Intelligent mode: project each successful execution's fact-occurrence vector onto
    the latest principal component and pick the top score (ties: artefacts count, then
    response length). Other modes: fall back to pairwise LLM selection.
    """
    if not executions:
        return executions

    for ex in executions:
        if ex.is_best:
            ex.is_best = False

    mode = (response_mode or "").strip().lower()
    if mode == "intelligent":
        # Require principal component from latest successful execution
        latest_pc_entry = None
        if successful_executions:
            try:
                latest_pc_entry = max(
                    (e for e in successful_executions if isinstance(e, dict) and e.get("principal_component_1")),
                    key=lambda e: e.get("order", -1),
                )
            except Exception:
                latest_pc_entry = None

        if not latest_pc_entry:
            logger.info("response_selector: No principal component available; falling back to LLM selection.")
        else:
            pc_facts = latest_pc_entry.get("principal_component_facts") or []
            pc_weights = latest_pc_entry.get("principal_component_1") or []
            if not pc_facts or not pc_weights or len(pc_facts) != len(pc_weights):
                logger.info("response_selector: Invalid principal component data; falling back to LLM selection.")
            else:
                fact_index = {num: idx for idx, num in enumerate(pc_facts) if isinstance(num, int)}
                def _exec_fact_indices(ex: Execution) -> List[int]:
                    if not successful_executions:
                        return []
                    matches = [
                        e for e in successful_executions
                        if isinstance(e, dict)
                        and e.get("branch_id") == ex.parallel_agent_id
                        and e.get("retry_number") == ex.retry_number
                    ]
                    if not matches:
                        return []
                    return matches[-1].get("response_fact_indices") or []

                def _response_text(ex: Execution) -> str:
                    if ex.final_response and isinstance(ex.final_response, AIMessage):
                        return _strip_material_icons(str(ex.final_response.content or ""))
                    return ""

                def _score(ex: Execution) -> tuple[float, int, int, str]:
                    indices = _exec_fact_indices(ex)
                    vec = [0.0] * len(pc_facts)
                    for num in indices:
                        if num in fact_index:
                            vec[fact_index[num]] = 1.0
                    score = sum(w * v for w, v in zip(pc_weights, vec))
                    artefact_count = len(ex.artefacts or [])
                    text = _response_text(ex)
                    return score, artefact_count, len(text), text

                candidates = [ex for ex in executions if ex.is_sufficient]
                if not candidates:
                    logger.info("response_selector: No successful executions; falling back to LLM selection.")
                else:
                    best = None
                    best_tuple = None
                    for ex in candidates:
                        score, artefacts_ct, text_len, text = _score(ex)
                        logger.info(
                            "[Intelligent Selector] Exec (branch=%d retry=%d) score=%0.4f artefacts=%d chars=%d text=%s",
                            ex.parallel_agent_id,
                            ex.retry_number,
                            score,
                            artefacts_ct,
                            text_len,
                            text,
                        )
                        key = (score, artefacts_ct, text_len)
                        if best is None or key > best_tuple:
                            best = ex
                            best_tuple = key
                    if best:
                        best.is_best = True
                        return executions

    # Fallback: existing LLM-based selection
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
