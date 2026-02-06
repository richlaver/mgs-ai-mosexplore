import ast
import asyncio
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from e2b_code_interpreter import AsyncSandbox

logger = logging.getLogger(__name__)

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")


def format_offset(delta: timedelta | None) -> str:
    if delta is None:
        return "unknown"
    total_minutes = int(delta.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    abs_minutes = abs(total_minutes)
    hours, minutes = divmod(abs_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def tzinfo_from_offset(offset_text: str | None) -> timezone | None:
    if not offset_text:
        return None
    cleaned = offset_text.strip().replace("UTC", "")
    match = _OFFSET_RE.match(cleaned)
    if not match:
        return None
    sign, hours, minutes = match.groups()
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    if sign == "-":
        delta = -delta
    return timezone(delta)


def describe_timezone(tz: timezone | ZoneInfo | None, *, label: str, source: str) -> dict:
    now = datetime.now(tz=tz or datetime.now().astimezone().tzinfo)
    tzinfo = now.tzinfo
    offset = format_offset(tzinfo.utcoffset(now)) if tzinfo else "unknown"
    tzname = tzinfo.key if isinstance(tzinfo, ZoneInfo) else (tzinfo.tzname(now) if tzinfo else "unknown")
    return {
        "label": label,
        "tzname": tzname,
        "offset": offset,
        "source": source,
        "display": f"{tzname or label} (UTC{offset})",
    }


def detect_user_timezone() -> dict:
    params = {}
    try:
        params = st.query_params or {}
    except Exception:
        params = {}
    candidates: list[str] = []
    for key in ("timezone", "tz"):
        value = params.get(key)
        if value and isinstance(value, list) and value[0]:
            candidates.append(value[0])
    env_tz = st.secrets.get("user_timezone") if hasattr(st, "secrets") else None
    if env_tz:
        candidates.insert(0, str(env_tz))

    for candidate in candidates:
        try:
            zone = ZoneInfo(candidate)
            return describe_timezone(zone, label=candidate, source="user-specified")
        except Exception:
            logger.warning("Invalid user timezone candidate ignored: %s", candidate)

    tzinfo = datetime.now().astimezone().tzinfo
    return describe_timezone(tzinfo, label="app-default", source="fallback-app-tz")


def fetch_project_timezone(db) -> dict:
    query = "SELECT timezone_diff FROM project_settings LIMIT 1"
    project_offset = None
    try:
        raw_result = db.run(query)
        parsed = ast.literal_eval(raw_result) if isinstance(raw_result, str) else raw_result
        if parsed and isinstance(parsed, (list, tuple)):
            row = parsed[0] if isinstance(parsed[0], (list, tuple)) else parsed[0].values() if isinstance(parsed[0], dict) else None
            if row:
                first_val = row[0] if isinstance(row, (list, tuple)) else list(row)[0]
                project_offset = str(first_val).strip()
    except Exception:
        logger.exception("Failed to fetch project timezone from project_settings")

    tzinfo = tzinfo_from_offset(project_offset) or timezone.utc
    offset = project_offset or format_offset(tzinfo.utcoffset(datetime.now(tzinfo)))
    return {
        "label": f"UTC{offset}",
        "tzname": tzinfo.tzname(datetime.now(tzinfo)) if tzinfo else "UTC",
        "offset": offset,
        "source": "project_settings" if project_offset else "default-utc",
        "display": f"UTC{offset}",
    }


def fetch_sandbox_timezone() -> dict:
    template_name = os.environ.get("E2B_TEMPLATE_NAME")
    if not template_name:
        logger.error("E2B_TEMPLATE_NAME not set; falling back to app timezone")
        fallback_tz = datetime.now().astimezone().tzinfo
        return describe_timezone(fallback_tz, label="sandbox-fallback", source="sandbox-fallback")

    sandbox_code = """
import datetime
import json

now = datetime.datetime.now().astimezone()
tzinfo = now.tzinfo
offset = tzinfo.utcoffset(now) if tzinfo else None
if offset is None:
    offset_str = "unknown"
else:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    offset_str = f"{sign}{hours:02d}:{minutes:02d}"

tzname = tzinfo.tzname(now) if tzinfo else "unknown"
print(json.dumps({"tzname": tzname, "offset": offset_str}))
"""

    async def _run() -> dict:
        sandbox = await AsyncSandbox.create(
            template=template_name,
            timeout=60,
            request_timeout=60,
        )
        try:
            execution = await sandbox.run_code(sandbox_code)
            logs = getattr(execution, "logs", None)
            stdout = getattr(logs, "stdout", "") if logs else ""
            if isinstance(stdout, list):
                stdout = "\n".join(str(item) for item in stdout)
            payload = None
            for line in (stdout or "").splitlines():
                try:
                    payload = json.loads(line.strip())
                except Exception:
                    continue
            if isinstance(payload, dict):
                tzname = str(payload.get("tzname") or "sandbox")
                offset = str(payload.get("offset") or "unknown")
                label = tzname or "sandbox"
                display = f"{label} (UTC{offset})" if offset != "unknown" else label
                return {
                    "label": label,
                    "tzname": tzname,
                    "offset": offset,
                    "source": "sandbox",
                    "display": display,
                }
        finally:
            await sandbox.kill()
        raise RuntimeError("Failed to read sandbox timezone")

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())

        result_holder: dict[str, dict] = {}

        def _thread_runner() -> None:
            result_holder["result"] = asyncio.run(_run())

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()
        if result_holder.get("result"):
            return result_holder["result"]
    except Exception:
        logger.exception("Failed to fetch sandbox timezone")

    fallback_tz = datetime.now().astimezone().tzinfo
    return describe_timezone(fallback_tz, label="sandbox-fallback", source="sandbox-fallback")


def init_timezones(db) -> None:
    app_tzinfo = datetime.now().astimezone().tzinfo
    app_tz = describe_timezone(app_tzinfo, label="app", source="app-server")
    user_tz = detect_user_timezone()
    sandbox_tz = fetch_sandbox_timezone()
    project_tz = fetch_project_timezone(db)

    st.session_state.timezone_context = {
        "app_timezone": app_tz.get("display"),
        "user_timezone": user_tz.get("display"),
        "sandbox_timezone": sandbox_tz.get("display"),
        "project_timezone": project_tz.get("display"),
        "project_timezone_offset": project_tz.get("offset"),
        "sources": {
            "app": app_tz.get("source"),
            "user": user_tz.get("source"),
            "sandbox": sandbox_tz.get("source"),
            "project": project_tz.get("source"),
        },
    }

    logger.info(
        "Timezone context initialised | app=%s user=%s sandbox=%s project=%s",
        st.session_state.timezone_context.get("app_timezone"),
        st.session_state.timezone_context.get("user_timezone"),
        st.session_state.timezone_context.get("sandbox_timezone"),
        st.session_state.timezone_context.get("project_timezone"),
    )
