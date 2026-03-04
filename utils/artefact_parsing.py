import json
import re
from typing import Any


UUID_V4_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)


def identify_plot_or_csv_and_extract_artefact_id(payload: Any) -> tuple[str | None, str | None]:
    try:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)

    plot_marker = '"type": "plot"'
    csv_marker = '"type": "csv"'
    plot_idx = text.find(plot_marker)
    csv_idx = text.find(csv_marker)

    yield_type: str | None = None
    if plot_idx != -1 and csv_idx != -1:
        yield_type = "plot" if plot_idx <= csv_idx else "csv"
    elif plot_idx != -1:
        yield_type = "plot"
    elif csv_idx != -1:
        yield_type = "csv"

    if yield_type is None:
        return None, None

    match = UUID_V4_RE.search(text)
    artefact_id = match.group(0) if match else None
    return yield_type, artefact_id