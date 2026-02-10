from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional

from e2b_sandbox import prewarm_sandbox

logger = logging.getLogger(__name__)


PrewarmLogCallback = Callable[[str, str, int], None]


def _prewarm_worker(
    *,
    slot_id: int,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    selected_project_key: Optional[str],
    log_callback: Optional[PrewarmLogCallback],
) -> None:
    def _log(message: str, level: str = "debug") -> None:
        if log_callback:
            try:
                log_callback(message, level, slot_id)
            except Exception:
                logger.debug("Prewarm log callback failed", exc_info=True)
        else:
            getattr(logger, level, logger.debug)("%s", message)

    _log(f"Sandbox prewarm start (branch={slot_id})", "debug")
    try:
        prewarm_sandbox(
            slot_id=slot_id,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            thread_id=thread_id,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access,
            selected_project_key=selected_project_key,
            log_callback=lambda msg, lvl="debug": _log(msg, lvl),
        )
    except Exception as exc:
        _log(f"Sandbox prewarm failed: {exc}", "error")
    else:
        _log(f"Sandbox prewarm ready (branch={slot_id})", "debug")


def start_sandbox_prewarm_threads(
    *,
    num_slots: int,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    selected_project_key: Optional[str] = None,
    log_callback: Optional[PrewarmLogCallback] = None,
) -> List[threading.Thread]:
    threads: List[threading.Thread] = []
    for slot_id in range(max(0, num_slots)):
        thread = threading.Thread(
            target=_prewarm_worker,
            name=f"sandbox-prewarm-{slot_id}",
            kwargs={
                "slot_id": slot_id,
                "table_info": table_info,
                "table_relationship_graph": table_relationship_graph,
                "thread_id": thread_id,
                "user_id": user_id,
                "global_hierarchy_access": global_hierarchy_access,
                "selected_project_key": selected_project_key,
                "log_callback": log_callback,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    return threads
