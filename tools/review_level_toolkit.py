import logging
import re
import json
import datetime as dt
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Type, Tuple
import pandas as pd

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import BaseTool
from tools.sql_security_toolkit import GeneralSQLQueryTool

logger = logging.getLogger(__name__)

_VALID_DB_FIELD_RE = re.compile(r"^(data|calculation)\d+$")

NO_DATA_MSG = "No data was found in the database matching the specified search criteria."

def _quote(val: Union[str, float, int, datetime, None]) -> str:
    """Return a safely quoted SQL literal (basic). For GeneralSQLQueryTool which accepts final SQL string only."""
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, datetime):
        return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
    return "'" + str(val).replace("'", "''") + "'"

def _validate_col(name: str) -> str:
    """Validate db_field_name limited to 'dataN' or 'calculationN' (N=digits)."""
    if not _VALID_DB_FIELD_RE.match(name):
        raise ValueError(f"Invalid db_field_name (must be dataN or calculationN): {name}")
    return name

def _is_no_data(result: Optional[str]) -> bool:
    """Return True if the tool returned the canonical no-data message or None.
    """
    if result is None:
        return True
    s = str(result).strip()
    return s == NO_DATA_MSG


def _parse_rows(result: Union[str, List[Dict[str, Any]], None]) -> List[Dict[str, Any]]:
    """Parse GeneralSQLQueryTool string output into a list of dicts.

    Supports JSON strings, Python repr with datetime/Decimal, or already-materialized lists.
    Returns [] when empty/no data.
    Raises ValueError on hard parse failures.
    """
    if result is None:
        return []
    if isinstance(result, list):
        return [dict(r) for r in result]
    text = str(result).strip()
    if _is_no_data(text):
        return []
    if text.startswith("Error:"):
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [dict(r) for r in parsed]
    except Exception:
        pass

    try:
        env = {"__builtins__": {}, "datetime": dt, "Decimal": Decimal}
        parsed = eval(text, env)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [dict(r) for r in parsed]
    except Exception as e:
        raise ValueError(f"parse failed: {e}")

    return []


def _coerce_float(val: Any) -> Optional[float]:
    """Best-effort conversion to float for DB values.

    Handles int, float, Decimal, strings (including double-quoted numerics like ""-7.37""),
    and returns None for null-like inputs.
    """
    if val is None:
        return None
    if isinstance(val, float):
        return val
    if isinstance(val, int):
        return float(val)
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1]
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    try:
        return float(val)
    except Exception:
        return None


def _coerce_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (float, Decimal)):
        try:
            return int(val)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1]
        try:
            return int(float(s))
        except Exception:
            return None
    try:
        return int(val)
    except Exception:
        return None

class _BaseQueryTool(BaseModel):
    """Base tool for review-level operations using SQL."""
    sql_tool: GeneralSQLQueryTool = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class _ReviewStatusOutput(BaseModel):
    """Output schema for review status with value and timestamp."""
    review_status: Optional[str] = Field(None, description="Name of the most severe breached review level, or None.")
    db_field_value: Optional[float] = Field(None, description="The database field value used for review check.")
    db_field_value_timestamp: Optional[datetime] = Field(None, description="Timestamp of the database field value.")


class _ReviewLevelSchema(BaseModel):
    """Schema for a single review level."""
    review_name: str = Field(..., description="User-facing name of the review level.")
    review_value: float = Field(..., description="Threshold value for this review level.")
    review_direction: str = Field(..., description="'upper' or 'lower' indicating breach direction.")
    review_color: str = Field(..., description="Hex color string with # prefix (e.g., '#FF0000').")


class _ReviewValueOutput(BaseModel):
    """Output schema for retrieving a specific review level's threshold."""
    review_value: float = Field(..., description="Threshold value for the specified review level.")
    review_direction: str = Field(..., description="'upper' or 'lower'.")


class _BreachedInstrumentReading(BaseModel):
    """Schema for a single breached instrument reading."""
    instrument_id: str = Field(..., description="Instrument ID (from instrum.instr_id).")
    db_field_name: str = Field(..., description="System field name (dataN or calculationN) that breached.")
    review_name: Optional[str] = Field(None, description="Name of the breached review level (most severe or specific).")
    field_value: float = Field(..., description="Most recent field value breaching the review level.")
    field_value_timestamp: datetime = Field(..., description="Timestamp of the field value.")
    review_value: float = Field(..., description="Threshold value of the named review level.")

class _ReviewStatusByValueItem(BaseModel):
    instrument_id: str = Field(..., description="Instrument ID")
    db_field_name: str = Field(..., description="System field name (dataN or calculationN)")
    db_field_value: float = Field(..., description="Measured field value to evaluate")

class _ReviewStatusByValueInput(BaseModel):
    items: List[_ReviewStatusByValueItem] = Field(..., description="List of review status value evaluation items")

class _ReviewStatusByTimeItem(BaseModel):
    instrument_id: str = Field(..., description="Instrument ID")
    db_field_name: str = Field(..., description="System field name (dataN or calculationN)")

class _ReviewStatusByTimeInput(BaseModel):
    items: List[_ReviewStatusByTimeItem] = Field(..., description="List of review status time evaluation items")
    timestamp: Union[str, datetime] = Field(..., description="Reference timestamp (ISO string or datetime)")

class _ReviewSchemaItem(BaseModel):
    instrument_id: str = Field(..., description="Instrument ID")
    db_field_name: str = Field(..., description="System field name (dataN or calculationN)")

class _ReviewSchemaInput(BaseModel):
    items: List[_ReviewSchemaItem] = Field(..., description="List of schema retrieval items")

class _ReviewValueItem(BaseModel):
    instrument_id: str = Field(..., description="Instrument ID")
    db_field_name: str = Field(..., description="System field name (dataN or calculationN)")
    review_name: str = Field(..., description="Review level name")

class _ReviewValueInput(BaseModel):
    items: List[_ReviewValueItem] = Field(..., description="List of review value retrieval items")


class GetReviewStatusByValueTool(BaseTool, _BaseQueryTool):
    """
    Returns the review level status (most severe breached review level name) for one or more database field values at specified instruments and fields. Returns None if no review is breached or no reviews exist.

    Input:
    List of dictionaries with keys:
    - instrument_id: str - The instrument ID (from instrum.instr_id)
    - db_field_name: str - The system field name (e.g., 'data1', 'calculation1')
    - db_field_value: float - The actual field value to evaluate against review thresholds

    Output:
    Either DataFrame with columns:
    - instrument_id: str
    - db_field_name: str
    - db_field_value: float
    - review_name: review level name if breached, NaN if not
    Or None if no active reviews found
    Or ERROR: message if invalid
    """
    name: str = "get_review_status_by_value_tool"
    description: str = (
        """
        Returns the review level status (most severe breached review level name) for one or more database field values at specified instruments and fields. Returns None if no review is breached or no reviews exist.

        Input:
        List of dictionaries with keys:
        - instrument_id: str - The instrument ID (from instrum.instr_id)
        - db_field_name: str - The system field name (e.g., 'data1', 'calculation1')
        - db_field_value: float - The actual field value to evaluate against review thresholds

        Output:
        Either Pandas DataFrame with columns:
        - instrument_id: str
        - db_field_name: str
        - db_field_value: float
        - review_name: review level name if breached, NaN if not
        Or None if no active reviews found
        Or ERROR: message if invalid
        """
    )
    args_schema: Type[BaseModel] = _ReviewStatusByValueInput

    def _run(self, items: List[_ReviewStatusByValueItem]) -> Union[pd.DataFrame, None, str]:
        inputs = items
        if not isinstance(inputs, list) or not inputs:
            return "ERROR: inputs must be a non-empty list of {instrument_id, db_field_name, db_field_value}."

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        try:
            for idx, item in enumerate(inputs):
                instr = item.get("instrument_id")
                field = item.get("db_field_name")
                val = item.get("db_field_value")
                if not instr or not field:
                    return "ERROR: instrument_id and db_field_name are required for each item."
                try:
                    _ = float(val)
                except (TypeError, ValueError):
                    return "ERROR: db_field_value must be numeric for each item."
                grouped.setdefault(field, []).append({"instrument_id": instr, "db_field_value": float(val)})
        except Exception as e:
            return f"ERROR: inputs validation failed: {e}"

        all_rows: List[Dict[str, Any]] = []
        for field, items in grouped.items():
            try:
                values_sql_parts = []
                for i, it in enumerate(items):
                    values_sql_parts.append(
                        "SELECT "
                        f"{i} AS param_id, "
                        f"{_quote(it['instrument_id'])} AS instr_id, "
                        f"{_quote(field)} AS review_field, "
                        f"CAST({_quote(it['db_field_value'])} AS DECIMAL(20,6)) AS field_value"
                    )
                params_cte = " WITH params AS ( " + " UNION ALL ".join(values_sql_parts) + " ) "
            except Exception as e:
                return f"ERROR: render parameters failed: {e}"

            sql = (
                params_cte +
                " , breaches AS ( "
                " SELECT p.param_id, rl.review_name, riv.review_direction, "
                " CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
                " ROW_NUMBER() OVER (PARTITION BY p.param_id ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END) AS severity_rank "
                " FROM params p "
                " JOIN review_instruments ri ON ri.instr_id = p.instr_id AND ri.review_field = p.review_field AND ri.review_status = 'ON' "
                " JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                " JOIN review_levels rl ON riv.review_level_id = rl.id "
                " WHERE REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                " AND ((riv.review_direction = 1 AND p.field_value > CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))) "
                " OR (riv.review_direction = -1 AND p.field_value < CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)))) "
                " ) "
                " SELECT p.instr_id AS instrument_id, p.review_field AS db_field_name, p.field_value AS db_field_value, b.review_name "
                " FROM params p "
                " LEFT JOIN (SELECT param_id, review_name FROM breaches WHERE severity_rank = 1) b ON b.param_id = p.param_id "
            )

            try:
                logger.info("[get_review_status_from_value/batch] SQL=%s", sql)
                result = self.sql_tool._run(sql)
                logger.info("[get_review_status_from_value/batch] Raw result=%s", result)
            except Exception as e:
                return f"ERROR: batch query failed: {e}"

            if _is_no_data(result):
                continue
            try:
                rows = _parse_rows(result)
            except Exception as e:
                return f"ERROR: batch parse failed: {e}"
            for r in rows:
                all_rows.append({
                    "instrument_id": r.get("instrument_id"),
                    "db_field_name": field,
                    "db_field_value": _coerce_float(r.get("db_field_value")),
                    "review_name": r.get("review_name"),
                })

        return pd.DataFrame(all_rows) if all_rows else None

class GetReviewStatusByTimeTool(BaseTool, _BaseQueryTool):
    """
    Finds the most recent reading before `timestamp` for one or more pairs of `instrument_id` and `db_field_name`, then for each pair returns its review status, value, and timestamp.

    Input:
    List of dictionaries with keys:
    - instrument_id: str - The instrument ID (from instrum.instr_id)
    - db_field_name: str - The database field name (e.g., 'dataN' or 'calculationN')
    In addition to list:
    - timestamp: ISO string or datetime

    Output:
    Either DataFrame with columns:
    - instrument_id: str
    - db_field_name: str
    - db_field_value: float
    - db_field_value_timestamp: datetime
    - review_name: str (NaN if none)
    Or None if no valid readings found
    Or ERROR: message if invalid
    """
    name: str = "get_review_status_by_time_tool"
    description: str = (
        """
        Finds the most recent reading before `timestamp` for one or more pairs of `instrument_id` and `db_field_name`, then for each pair returns its review status, value, and timestamp.

        Input:
        List of dictionaries with keys:
        - instrument_id: str - The instrument ID (from instrum.instr_id)
        - db_field_name: str - The database field name (e.g., 'dataN' or 'calculationN')
        In addition to list:
        - timestamp: ISO string or datetime

        Output:
        Either DataFrame with columns:
        - instrument_id: str
        - db_field_name: str
        - db_field_value: float
        - db_field_value_timestamp: datetime
        - review_name: str (NaN if none)
        Or None if no valid readings found
        Or ERROR: message if invalid
        """
    )

    args_schema: Type[BaseModel] = _ReviewStatusByTimeInput

    def _run(self, items: List[_ReviewStatusByTimeItem], timestamp: Union[str, datetime]) -> Union[pd.DataFrame, None, str]:
        if not isinstance(items, list) or not items:
            return "ERROR: items must be a non-empty list of {instrument_id, db_field_name}."
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                return "ERROR: Invalid timestamp format (use ISO)."

        grouped: Dict[str, List[str]] = {}
        try:
            for it in items:
                instr = it.get("instrument_id")
                field = it.get("db_field_name")
                if not instr or not field:
                    return "ERROR: instrument_id and db_field_name are required for each item."
                grouped.setdefault(field, []).append(instr)
        except Exception as e:
            return f"ERROR: items validation failed: {e}"

        out_rows: List[Dict[str, Any]] = []
        for field, instr_list in grouped.items():
            instr_list = list(dict.fromkeys(instr_list))
            try:
                values_sql_parts = []
                for i, instr in enumerate(instr_list):
                    values_sql_parts.append(
                        "SELECT "
                        f"{i} AS rn, {_quote(instr)} AS instr_id"
                    )
                params_cte = "WITH params AS ( " + " UNION ALL ".join(values_sql_parts) + " ) "
            except Exception as e:
                return f"ERROR: render parameters failed: {e}"

            if field.startswith("calculation"):
                json_path = f"$.$FIELD$"
                json_path = json_path.replace("$FIELD$", field)
                field_expr = (
                    "CASE WHEN JSON_VALID(m.custom_fields) "
                    f"AND JSON_EXTRACT(m.custom_fields, {_quote(json_path)}) IS NOT NULL "
                    f"AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, {_quote(json_path)}))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                    f"THEN CAST(REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, {_quote(json_path)}))), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, {_quote(json_path)}) IS NOT NULL "
                    f"AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, {_quote(json_path)}))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
                )
                valid_col = field
            else:
                try:
                    valid_col = _validate_col(field)
                except ValueError as e:
                    return f"ERROR: {e}"
                field_expr = (
                    f"CASE WHEN REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                    f"THEN CAST(REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
                )

            sql = (
                params_cte +
                " , latest_reading AS ( "
                " SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value, "
                " ROW_NUMBER() OVER (PARTITION BY m.instr_id ORDER BY m.date1 DESC) AS rnk "
                " FROM mydata m JOIN params p ON p.instr_id = m.instr_id "
                f" WHERE m.date1 < {_quote(timestamp)} AND " + reading_is_not_null + " ), "
                " chosen AS ( SELECT instr_id, reading_time, field_value FROM latest_reading WHERE rnk = 1 ), "
                " reviews AS ( "
                " SELECT ri.instr_id, rl.review_name, "
                " CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
                " riv.review_direction "
                " FROM review_instruments ri "
                " JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                " JOIN review_levels rl ON riv.review_level_id = rl.id "
                f" WHERE ri.review_field = {_quote(valid_col)} AND ri.review_status = 'ON' ), "
                " breaches AS ( "
                " SELECT c.instr_id, c.field_value, c.reading_time, r.review_name, r.review_value, r.review_direction, "
                " ROW_NUMBER() OVER (PARTITION BY c.instr_id ORDER BY CASE WHEN r.review_direction = 1 THEN -r.review_value ELSE r.review_value END) AS severity_rank "
                " FROM chosen c JOIN reviews r ON r.instr_id = c.instr_id "
                " WHERE ((r.review_direction = 1 AND c.field_value > r.review_value) OR (r.review_direction = -1 AND c.field_value < r.review_value)) ), "
                " best AS ( SELECT instr_id, review_name FROM breaches WHERE severity_rank = 1 ) "
                " SELECT c.instr_id AS instrument_id, " + _quote(valid_col) + " AS db_field_name, c.field_value AS db_field_value, c.reading_time AS db_field_value_timestamp, b.review_name "
                " FROM chosen c LEFT JOIN best b ON b.instr_id = c.instr_id "
            )

            try:
                logger.info("[get_review_status_from_time/batch] SQL=%s", sql)
                result = self.sql_tool._run(sql)
                logger.info("[get_review_status_from_time/batch] Raw result=%s", result)
            except Exception as e:
                return f"ERROR: batch query failed: {e}"
            if _is_no_data(result):
                continue
            try:
                rows = _parse_rows(result)
            except Exception as e:
                return f"ERROR: batch parse failed: {e}"
            for r in rows:
                fv = _coerce_float(r.get("db_field_value"))
                if fv is None:
                    # Skip invalid reading
                    continue
                out_rows.append({
                    "instrument_id": r.get("instrument_id"),
                    "db_field_name": field,
                    "review_name": r.get("review_name"),
                    "db_field_value": fv,
                    "db_field_value_timestamp": r.get("db_field_value_timestamp"),
                })

        return pd.DataFrame(out_rows) if out_rows else None

class GetReviewSchemaTool(BaseTool, _BaseQueryTool):
    """
    Returns all active review levels for one or more instruments for one or more fields, including name, value, direction, and color.

    Input:
    List of dictionaries with keys:
    - instrument_id: str - The instrument ID (from instrum.instr_id)
    - db_field_name: str - The system field name (e.g., 'dataN', 'calculationN')

    Output:
    Either DataFrame with columns:
    - instrument_id: str
    - db_field_name: str
    - review_name: str
    - review_value: float
    - review_direction: str ('upper' or 'lower')
    - review_color: str (hex color with # prefix)
    Or None if no active review levels found
    Or ERROR: message if invalid
    """
    name: str = "get_review_schema_tool"
    description: str = (
        """
        Returns all active review levels for one or more instruments for one or more fields, including name, value, direction, and color.

        Input:
        List of dictionaries with keys:
        - instrument_id: str - The instrument ID (from instrum.instr_id)
        - db_field_name: str - The system field name (e.g., 'dataN', 'calculationN')

        Output:
        Either DataFrame with columns:
        - instrument_id: str
        - db_field_name: str
        - review_name: str
        - review_value: float
        - review_direction: str ('upper' or 'lower')
        - review_color: str (hex color with # prefix)
        Or None if no active review levels found
        Or ERROR: message if invalid
        """
    )

    args_schema: Type[BaseModel] = _ReviewSchemaInput

    def _run(self, items: List[_ReviewSchemaItem]) -> Union[pd.DataFrame, None, str]:
        if not isinstance(items, list) or not items:
            return "ERROR: items must be a non-empty list of {instrument_id, db_field_name}."

        logger.info("[get_review_schema] Starting run with %d item(s)", len(items))

        grouped: Dict[str, List[str]] = {}
        for it in items:
            instr = it.get("instrument_id")
            field = it.get("db_field_name")
            if not instr or not field:
                return "ERROR: instrument_id and db_field_name are required for each item."
            grouped.setdefault(field, []).append(instr)

        logger.info(
            "[get_review_schema] Grouped into %d distinct field(s): %s",
            len(grouped),
            ", ".join(grouped.keys()) or "<none>"
        )

        all_rows: List[Dict[str, Any]] = []
        for field, instr_list in grouped.items():
            instr_list = list(dict.fromkeys(instr_list))
            logger.info(
                "[get_review_schema/%s] Preparing batch for %d instrument(s): %s",
                field,
                len(instr_list),
                ", ".join(instr_list)
            )
            values_sql_parts = [
                f"SELECT {i} AS rn, {_quote(instr)} AS instr_id" for i, instr in enumerate(instr_list)
            ]
            params_cte = "WITH params AS ( " + " UNION ALL ".join(values_sql_parts) + " ) "

            sql = (
                params_cte +
                " SELECT ri.instr_id AS instrument_id, " + _quote(field) + " AS db_field_name, rl.review_name, "
                " CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
                " CASE WHEN riv.review_direction = 1 THEN 'upper' ELSE 'lower' END AS review_direction, "
                " CONCAT('#', aci.aaa_color) AS review_color "
                " FROM params p "
                " JOIN review_instruments ri ON ri.instr_id = p.instr_id AND ri.review_status = 'ON' AND ri.review_field = " + _quote(field) +
                " JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                " JOIN review_levels rl ON riv.review_level_id = rl.id "
                " JOIN aaa_color_info aci ON rl.id = aci.review_id "
                " ORDER BY aci.`order`"
            )

            try:
                logger.info("[get_review_schema/batch] SQL=%s", sql)
                result = self.sql_tool._run(sql)
                logger.info("[get_review_schema/batch] Raw result=%s", result)
            except Exception as e:
                return f"ERROR: schema batch query failed: {e}"
            if _is_no_data(result):
                logger.info("[get_review_schema/%s] Result indicated no data", field)
                continue
            try:
                rows = _parse_rows(result)
                logger.info("[get_review_schema/%s] Parsed %d row(s)", field, len(rows))
            except Exception as e:
                return f"ERROR: schema batch parse failed: {e}"
            for r in rows:
                all_rows.append({
                    "instrument_id": r.get("instrument_id"),
                    "db_field_name": field,
                    "review_name": r.get("review_name"),
                    "review_value": _coerce_float(r.get("review_value")) if r.get("review_value") is not None else None,
                    "review_direction": r.get("review_direction"),
                    "review_color": r.get("review_color"),
                })

        logger.info(
            "[get_review_schema] Returning %s",
            f"DataFrame with {len(all_rows)} row(s)" if all_rows else "None"
        )

        return pd.DataFrame(all_rows) if all_rows else None

class GetReviewValueTool(BaseTool, _BaseQueryTool):
    """
    Given combinations of instrument_id, db_field_name, and review_name, for each combination returns the threshold value and direction.

    Input:
    List of dictionaries with keys:
    - instrument_id: str - The instrument ID (from instrum.instr_id)
    - db_field_name: str - The system field name (e.g., 'dataN', 'calculationN')
    - review_name: str - The name of the review level

    Output:
    Either DataFrame with columns:
    - instrument_id: str
    - db_field_name: str
    - review_name: str
    - review_value: float
    - review_direction: str ('upper' or 'lower')
    Or None if no matches found
    Or ERROR: message if invalid
    """
    name: str = "get_review_value_tool"
    description: str = (
        """
        Given combinations of instrument_id, db_field_name, and review_name, for each combination returns the threshold value and direction.

        Input:
        List of dictionaries with keys:
        - instrument_id: str - The instrument ID (from instrum.instr_id)
        - db_field_name: str - The system field name (e.g., 'dataN', 'calculationN')
        - review_name: str - The name of the review level

        Output:
        Either DataFrame with columns:
        - instrument_id: str
        - db_field_name: str
        - review_name: str
        - review_value: float
        - review_direction: str ('upper' or 'lower')
        Or None if no matches found
        Or ERROR: message if invalid
        """
    ) 

    args_schema: Type[BaseModel] = _ReviewValueInput

    def _run(self, items: List[_ReviewValueItem]) -> Union[pd.DataFrame, None, str]:
        if not isinstance(items, list) or not items:
            return "ERROR: items must be a non-empty list of {instrument_id, db_field_name, review_name}."

        grouped: Dict[str, List[Dict[str, str]]] = {}
        for it in items:
            instr = it.get("instrument_id")
            field = it.get("db_field_name")
            rname = it.get("review_name")
            if not instr or not field or not rname:
                return "ERROR: instrument_id, db_field_name, and review_name are required for each item."
            grouped.setdefault(field, []).append({"instrument_id": instr, "review_name": rname})

        out_rows: List[Dict[str, Any]] = []
        for field, tuples in grouped.items():
            values_sql_parts = []
            for i, t in enumerate(tuples):
                values_sql_parts.append(
                    "SELECT "
                    f"{i} AS rn, {_quote(t['instrument_id'])} AS instr_id, {_quote(t['review_name'])} AS review_name"
                )
            params_cte = "WITH params AS ( " + " UNION ALL ".join(values_sql_parts) + " ) "

            sql = (
                params_cte +
                " SELECT p.instr_id AS instrument_id, " + _quote(field) + " AS db_field_name, p.review_name, "
                " CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
                " CASE WHEN riv.review_direction = 1 THEN 'upper' ELSE 'lower' END AS review_direction "
                " FROM params p "
                " JOIN review_instruments ri ON ri.instr_id = p.instr_id AND ri.review_field = " + _quote(field) + " AND ri.review_status = 'ON' "
                " JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                " JOIN review_levels rl ON riv.review_level_id = rl.id AND rl.review_name = p.review_name "
                " WHERE (ri.effective_from IS NULL OR ri.effective_from <= NOW()) "
            )

            try:
                logger.info("[get_review_value/batch] SQL=%s", sql)
                result = self.sql_tool._run(sql)
                logger.info("[get_review_value/batch] Raw result=%s", result)
            except Exception as e:
                return f"ERROR: value batch query failed: {e}"
            if _is_no_data(result):
                continue
            try:
                rows = _parse_rows(result)
            except Exception as e:
                return f"ERROR: value batch parse failed: {e}"
            for r in rows:
                val = _coerce_float(r.get("review_value"))
                if val is None:
                    continue
                out_rows.append({
                    "instrument_id": r.get("instrument_id"),
                    "db_field_name": field,
                    "review_name": r.get("review_name"),
                    "review_value": val,
                    "review_direction": r.get("review_direction"),
                })

        return pd.DataFrame(out_rows) if out_rows else None


class GetBreachedInstrumentsTool(BaseTool, _BaseQueryTool):
    """
    Finds instruments (optionally filtered by type/subtype) where the latest reading before `timestamp` breaches the named review level, but does NOT breach any more severe level.
    Omitting `db_field_name` looks for reviews on all fields for instrument types and subtypes as specified by the input filters.
    Omitting `review_name` finds breaches across all review levels subject to other input specifications.

    Returns a pandas DataFrame with one row per breached instrument (columns match BreachedInstrumentReading), or None, or an ERROR.
    """
    name: str = "get_breached_instruments_tool"
    description: str = (
        """
        Finds instruments (optionally filtered by type/subtype) where the latest reading before `timestamp` breaches the named review level, but does NOT breach any more severe level.
        Omitting `db_field_name` looks for reviews on all fields for instrument types and subtypes as specified by the input filters.
        Omitting `review_name` finds breaches across all review levels subject to other input specifications.

        Returns a pandas DataFrame with one row per breached instrument (columns match BreachedInstrumentReading), or None, or an ERROR.
        """
    )

    def _run(
        self,
        review_name: Optional[str],
        instrument_type: Optional[str],
        instrument_subtype: Optional[str],
        db_field_name: Optional[str],
        timestamp: Union[str, datetime],
    ) -> Union[pd.DataFrame, None, str]:
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                return "ERROR: Invalid timestamp format. Use ISO format."

        instrument_type = instrument_type or None
        instrument_subtype = instrument_subtype or None

        def _result_join_and_where(row_alias: str = "r") -> Tuple[str, str]:
            join_clause = ""
            where_parts = ["b.severity_rank = 1"]
            if review_name:
                where_parts.append("b.review_name = %(review_name)s")
            if instrument_type or instrument_subtype:
                join_clause = f"JOIN instrum i ON {row_alias}.instr_id = i.instr_id "
                if instrument_type:
                    where_parts.append("i.type1 = %(instrument_type)s")
                if instrument_subtype:
                    where_parts.append("i.subtype1 = %(instrument_subtype)s")
            return join_clause, "WHERE " + " AND ".join(where_parts)

        def _discover_join_and_where() -> Tuple[str, str]:
            join_clause = ""
            where_parts = ["ri.review_status = 'ON'"]
            if instrument_type or instrument_subtype:
                join_clause = "JOIN instrum i ON ri.instr_id = i.instr_id "
                if instrument_type:
                    where_parts.append("i.type1 = %(instrument_type)s")
                if instrument_subtype:
                    where_parts.append("i.subtype1 = %(instrument_subtype)s")
            return join_clause, "WHERE " + " AND ".join(where_parts)
        if db_field_name:
            try:
                valid_col = _validate_col(db_field_name)
            except ValueError as e:
                return f"ERROR: {e}"

            if valid_col.startswith("calculation"):
                json_path = f"$.{valid_col}"
                field_expr = (
                    "CASE WHEN JSON_VALID(m.custom_fields) "
                    "AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                    "THEN CAST(REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') AS DECIMAL(20,6)) "
                    "ELSE NULL END"
                )
                reading_is_not_null = (
                    "JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
                )
            else:
                field_expr = (
                    f"CASE WHEN REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                    f"THEN CAST(REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
                )

            result_join, result_where = _result_join_and_where("r")

            sql = (
                "WITH latest_reading AS ( "
                "SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value "
                "FROM mydata m "
                "JOIN review_instruments ri ON m.instr_id = ri.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                "WHERE m.date1 < %(ts)s AND " + reading_is_not_null + " ), "
                "ranked AS ( "
                "SELECT instr_id, reading_time, field_value, ROW_NUMBER() OVER (PARTITION BY instr_id ORDER BY reading_time DESC) AS rn "
                "FROM latest_reading ), "
                "active_reviews AS ( "
                "SELECT ri.instr_id, rl.review_name, "
                "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value_clean, "
                "riv.review_direction, "
                "ROW_NUMBER() OVER (PARTITION BY ri.instr_id ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END) AS severity_rank "
                "FROM review_instruments ri "
                "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "JOIN review_levels rl ON riv.review_level_id = rl.id "
                "WHERE ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                "AND REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' ), "
                "breaches AS ( "
                "SELECT r.instr_id, r.field_value, r.reading_time, ar.review_name, ar.review_value_clean, ar.review_direction, ar.severity_rank "
                "FROM ranked r "
                "JOIN active_reviews ar ON ar.instr_id = r.instr_id "
                "WHERE r.rn = 1 AND ((ar.review_direction = 1 AND r.field_value > ar.review_value_clean) OR (ar.review_direction = -1 AND r.field_value < ar.review_value_clean)) ) "
                "SELECT r.instr_id AS instrument_id, %(review_field)s AS db_field_name, b.review_name, r.field_value, r.reading_time AS field_value_timestamp, b.review_value_clean AS review_value "
                "FROM ranked r "
                "JOIN breaches b ON b.instr_id = r.instr_id AND r.rn = 1 "
                + result_join +
                " " + result_where +
                " ORDER BY r.reading_time DESC"
            )

            params: Dict[str, Any] = {
                "instrument_type": instrument_type,
                "instrument_subtype": instrument_subtype,
                "review_field": valid_col,
                "ts": timestamp,
            }
            if review_name:
                params["review_name"] = review_name
            if valid_col.startswith("calculation"):
                params["json_path"] = json_path

            try:
                rendered = sql % {k: _quote(v) for k, v in params.items()}
            except Exception as e:
                return f"ERROR: render failed: {e}"
            try:
                logger.info("[get_breached_instruments] SQL=%s", rendered)
                result = self.sql_tool._run(rendered)
                logger.info("[get_breached_instruments] Raw result=%s", result)
            except Exception as e:
                return f"ERROR: unified query failed: {e}"
            if _is_no_data(result):
                return None
            try:
                rows = _parse_rows(result)
            except Exception as e:
                return f"ERROR: unified parse failed: {e}"
            if not rows:
                return None
            out: List[Dict[str, Any]] = []
            for r in rows:
                try:
                    fv = _coerce_float(r.get("field_value"))
                    rv = _coerce_float(r.get("review_value"))
                    if fv is None or rv is None:
                        raise ValueError("numeric conversion failed")
                    out.append(_BreachedInstrumentReading(
                        instrument_id=r.get("instrument_id"),
                        db_field_name=valid_col,
                        review_name=r.get("review_name"),
                        field_value=fv,
                        field_value_timestamp=r.get("field_value_timestamp"),
                        review_value=rv,
                    ).model_dump())
                except Exception as e:
                    logger.warning("[get_breached_instruments] Row parse skipped: %s row=%s", e, r)
            logger.info("[get_breached_instruments] Parsed %d breached instruments", len(out))
            return pd.DataFrame(out) if out else None

        discover_join, discover_where = _discover_join_and_where()
        discover_sql = (
            "SELECT DISTINCT ri.review_field AS db_field_name "
            "FROM review_instruments ri "
            + discover_join +
            " " + discover_where
        )
        try:
            rendered_discover = discover_sql % {
                "instrument_type": _quote(instrument_type),
                "instrument_subtype": _quote(instrument_subtype),
            }
        except Exception as e:
            return f"ERROR: render discover failed: {e}"

        try:
            logger.info("[get_breached_instruments/discover] SQL=%s", rendered_discover)
            discover_result = self.sql_tool._run(rendered_discover)
            logger.info("[get_breached_instruments/discover] Raw result=%s", discover_result)
        except Exception as e:
            return f"ERROR: discover query failed: {e}"

        if _is_no_data(discover_result):
            return None
        try:
            field_rows = _parse_rows(discover_result)
        except Exception as e:
            return f"ERROR: discover parse failed: {e}"

        fields: List[str] = []
        for r in field_rows:
            f = r.get("db_field_name")
            if not f:
                continue
            try:
                fields.append(_validate_col(str(f)))
            except ValueError:
                # Skip invalid field names defensively
                continue

        # Deduplicate and short-circuit if none
        fields = list(dict.fromkeys(fields))
        if not fields:
            return None

        aggregated: List[Dict[str, Any]] = []
        for valid_col in fields:
            if valid_col.startswith("calculation"):
                json_path = f"$.{valid_col}"
                field_expr = (
                    "CASE WHEN JSON_VALID(m.custom_fields) "
                    "AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                    "THEN CAST(REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') AS DECIMAL(20,6)) "
                    "ELSE NULL END"
                )
                reading_is_not_null = (
                    "JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
                )
            else:
                field_expr = (
                    f"CASE WHEN REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                    f"THEN CAST(REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
                )

            result_join, result_where = _result_join_and_where("r")

            sql = (
                "WITH latest_reading AS ( "
                "SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value "
                "FROM mydata m "
                "JOIN review_instruments ri ON m.instr_id = ri.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                "WHERE m.date1 < %(ts)s AND " + reading_is_not_null + " ), "
                "ranked AS ( "
                "SELECT instr_id, reading_time, field_value, ROW_NUMBER() OVER (PARTITION BY instr_id ORDER BY reading_time DESC) AS rn "
                "FROM latest_reading ), "
                "active_reviews AS ( "
                "SELECT ri.instr_id, rl.review_name, "
                "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value_clean, "
                "riv.review_direction, "
                "ROW_NUMBER() OVER (PARTITION BY ri.instr_id ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END) AS severity_rank "
                "FROM review_instruments ri "
                "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "JOIN review_levels rl ON riv.review_level_id = rl.id "
                "WHERE ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                "AND REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' ), "
                "breaches AS ( "
                "SELECT r.instr_id, r.field_value, r.reading_time, ar.review_name, ar.review_value_clean, ar.review_direction, ar.severity_rank "
                "FROM ranked r "
                "JOIN active_reviews ar ON ar.instr_id = r.instr_id "
                "WHERE r.rn = 1 AND ((ar.review_direction = 1 AND r.field_value > ar.review_value_clean) OR (ar.review_direction = -1 AND r.field_value < ar.review_value_clean)) ) "
                "SELECT r.instr_id AS instrument_id, %(review_field)s AS db_field_name, b.review_name, r.field_value, r.reading_time AS field_value_timestamp, b.review_value_clean AS review_value "
                "FROM ranked r "
                "JOIN breaches b ON b.instr_id = r.instr_id AND r.rn = 1 "
                + result_join +
                " " + result_where +
                " ORDER BY r.reading_time DESC"
            )

            params: Dict[str, Any] = {
                "instrument_type": instrument_type,
                "instrument_subtype": instrument_subtype,
                "review_field": valid_col,
                "ts": timestamp,
            }
            if review_name:
                params["review_name"] = review_name
            if valid_col.startswith("calculation"):
                params["json_path"] = json_path

            try:
                rendered = sql % {k: _quote(v) for k, v in params.items()}
            except Exception as e:
                logger.warning("[get_breached_instruments/field] Render failed for %s: %s", valid_col, e)
                continue
            try:
                logger.info("[get_breached_instruments/field] SQL(%s)=%s", valid_col, rendered)
                result = self.sql_tool._run(rendered)
                logger.info("[get_breached_instruments/field] Raw result=%s", result)
            except Exception as e:
                logger.warning("[get_breached_instruments/field] Query failed for %s: %s", valid_col, e)
                continue
            if _is_no_data(result):
                continue
            try:
                rows = _parse_rows(result)
            except Exception as e:
                logger.warning("[get_breached_instruments/field] Parse failed for %s: %s", valid_col, e)
                continue
            for r in rows:
                try:
                    fv = _coerce_float(r.get("field_value"))
                    rv = _coerce_float(r.get("review_value"))
                    if fv is None or rv is None:
                        raise ValueError("numeric conversion failed")
                    aggregated.append(_BreachedInstrumentReading(
                        instrument_id=r.get("instrument_id"),
                        db_field_name=valid_col,
                        review_name=r.get("review_name"),
                        field_value=fv,
                        field_value_timestamp=r.get("field_value_timestamp"),
                        review_value=rv,
                    ).model_dump())
                except Exception as e:
                    logger.warning("[get_breached_instruments/field] Row parse skipped for %s: %s row=%s", valid_col, e, r)

        logger.info("[get_breached_instruments] Aggregated %d breached instruments across %d fields", len(aggregated), len(fields))
        return pd.DataFrame(aggregated) if aggregated else None


class _MapSeriesReviewStatusRow(BaseModel):
    """Schema for map series review status output per instrument."""
    instrument_id: str
    easting: float
    northing: float
    review_status: Optional[str] = None
    review_value: Optional[float] = None
    db_field_value: float
    db_field_value_timestamp: datetime


class _MapSeriesReviewChangeRow(BaseModel):
    """Schema for map series review change output per instrument."""
    instrument_id: str
    easting: float
    northing: float
    start_review_status: Optional[str] = None
    start_review_value: Optional[float] = None
    start_db_field_value: float
    start_db_field_value_timestamp: datetime
    end_review_status: Optional[str] = None
    end_review_value: Optional[float] = None
    end_db_field_value: float
    end_db_field_value_timestamp: datetime


class GetMapSeriesReviewSnapshotTool(BaseTool, _BaseQueryTool):
    """
    For instruments of a given type/subtype within an easting/northing window, returns the most recent valid value before `timestamp` for `db_field_name` and the most severe breached review status for that instrument/field.

    Output columns:
    - instrument_id
    - easting
    - northing
    - review_status (can be NULL if no breach)
    - review_value (threshold of the returned status)
    - db_field_value
    - db_field_value_timestamp
    """

    name: str = "get_map_series_review_snapshot_tool"
    description: str = (
        """
        Map series review status for instruments filtered by type/subtype, easting/northing bounds, and an exclusion list. Supports system fields dataN and calculationN. Uses the most recent valid reading before the provided timestamp and returns the most severe breached review level per instrument (or NULL when none).
        """
    )

    def _run(
        self,
        instrument_type: str,
        instrument_subtype: Optional[str],
        db_field_name: str,
        timestamp: Union[str, datetime],
        min_e: float,
        max_e: float,
        min_n: float,
        max_n: float,
        exclude_instrument_ids: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, None, str]:
        if not instrument_type or not db_field_name:
            return "ERROR: instrument_type and db_field_name are required."

        try:
            valid_col = _validate_col(db_field_name)
        except ValueError as e:
            return f"ERROR: {e}"

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                return "ERROR: Invalid timestamp format (use ISO)."

        try:
            min_e = float(min_e)
            max_e = float(max_e)
            min_n = float(min_n)
            max_n = float(max_n)
        except Exception:
            return "ERROR: min/max easting/northing must be numeric."

        if valid_col.startswith("calculation"):
            json_path = f"$.{valid_col}"
            field_expr = (
                "CASE WHEN JSON_VALID(m.custom_fields) "
                "AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "THEN CAST(REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') AS DECIMAL(20,6)) "
                "ELSE NULL END"
            )
            reading_is_not_null = (
                "JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
            )
        else:
            field_expr = (
                f"CASE WHEN REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                f"THEN CAST(REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
            )
            reading_is_not_null = (
                f"REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
            )

        exclude_clause = ""
        if exclude_instrument_ids:
            try:
                in_list = ", ".join(_quote(x) for x in exclude_instrument_ids if x)
                if in_list:
                    exclude_clause = f" AND i.instr_id NOT IN ({in_list}) "
            except Exception as e:
                return f"ERROR: Failed to process exclude_instrument_ids: {e}"

        sql = (
            "WITH eligible AS ( "
            "SELECT i.instr_id, "
            "CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) AS easting, "
            "CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) AS northing "
            "FROM instrum i JOIN location l ON i.location_id = l.id "
            "WHERE i.type1 = %(instrument_type)s "
            "AND (%(instrument_subtype)s IS NULL OR i.subtype1 = %(instrument_subtype)s) "
            "AND REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "AND REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "AND CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_e)s AND %(max_e)s "
            "AND CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_n)s AND %(max_n)s "
            + exclude_clause +
            "), latest_reading AS ( "
            "SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value, "
            "ROW_NUMBER() OVER (PARTITION BY m.instr_id ORDER BY m.date1 DESC) AS rn "
            "FROM mydata m JOIN eligible e ON e.instr_id = m.instr_id "
            "WHERE m.date1 < %(ts)s AND " + reading_is_not_null + " ), "
            "chosen AS ( SELECT instr_id, reading_time, field_value FROM latest_reading WHERE rn = 1 ), "
            "reviews AS ( "
            "SELECT ri.instr_id, rl.review_name, "
            "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
            "riv.review_direction "
            "FROM review_instruments ri "
            "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "WHERE ri.review_field = %(review_field)s AND ri.review_status = 'ON' ), "
            "breaches AS ( "
            "SELECT c.instr_id, c.field_value, c.reading_time, r.review_name, r.review_value, r.review_direction, "
            "ROW_NUMBER() OVER (PARTITION BY c.instr_id ORDER BY CASE WHEN r.review_direction = 1 THEN -r.review_value ELSE r.review_value END) AS severity_rank "
            "FROM chosen c JOIN reviews r ON r.instr_id = c.instr_id "
            "WHERE ((r.review_direction = 1 AND c.field_value > r.review_value) OR (r.review_direction = -1 AND c.field_value < r.review_value)) ) "
            "SELECT e.instr_id AS instrument_id, e.easting, e.northing, b.review_name AS review_status, b.review_value, "
            "c.field_value AS db_field_value, c.reading_time AS db_field_value_timestamp "
            "FROM eligible e JOIN chosen c ON c.instr_id = e.instr_id "
            "LEFT JOIN (SELECT instr_id, review_name, review_value FROM breaches WHERE severity_rank = 1) b ON b.instr_id = e.instr_id "
            "ORDER BY c.reading_time DESC"
        )

        params: Dict[str, Any] = {
            "instrument_type": instrument_type,
            "instrument_subtype": instrument_subtype,
            "review_field": valid_col,
            "ts": timestamp,
            "min_e": min_e,
            "max_e": max_e,
            "min_n": min_n,
            "max_n": max_n,
        }
        if valid_col.startswith("calculation"):
            params["json_path"] = json_path

        try:
            rendered = sql % {k: _quote(v) for k, v in params.items()}
        except Exception as e:
            return f"ERROR: render failed: {e}"

        try:
            logger.info("[get_map_series_review_status] SQL=%s", rendered)
            result = self.sql_tool._run(rendered)
            logger.info("[get_map_series_review_status] Raw result=%s", result)
        except Exception as e:
            return f"ERROR: unified query failed: {e}"

        if _is_no_data(result):
            return None
        try:
            rows = _parse_rows(result)
        except Exception as e:
            return f"ERROR: unified parse failed: {e}"

        if not rows:
            return None

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                e = _coerce_float(r.get("easting"))
                n = _coerce_float(r.get("northing"))
                fv = _coerce_float(r.get("db_field_value"))
                rv = _coerce_float(r.get("review_value")) if r.get("review_value") is not None else None
                if e is None or n is None or fv is None:
                    raise ValueError("required numeric conversion failed")
                out.append(_MapSeriesReviewStatusRow(
                    instrument_id=r.get("instrument_id"),
                    easting=e,
                    northing=n,
                    review_status=r.get("review_status"),
                    review_value=rv,
                    db_field_value=fv,
                    db_field_value_timestamp=r.get("db_field_value_timestamp"),
                ).model_dump())
            except Exception as e:
                logger.warning("[get_map_series_review_status] Row skipped: %s row=%s", e, r)

        return pd.DataFrame(out) if out else None


class GetMapSeriesReviewChangeTool(BaseTool, _BaseQueryTool):
    """
    For instruments of a given type/subtype within an easting/northing window, compares the most recent valid values
    before `start_timestamp` and `end_timestamp` for `db_field_name`, then returns rows where the most severe breached
    review level changed between these two times (including transitions to/from no breach).

    Output columns:
    - instrument_id
    - easting
    - northing
    - start_review_status (can be NULL if no breach)
    - start_review_value (threshold of the returned status)
    - start_db_field_value
    - start_db_field_value_timestamp
    - end_review_status (can be NULL if no breach)
    - end_review_value (threshold of the returned status)
    - end_db_field_value
    - end_db_field_value_timestamp
    """

    name: str = "get_map_series_review_change_tool"
    description: str = (
        """
        Map series review change: filter instruments by type/subtype, easting/northing bounds, and optional exclusion list.
        Supports system fields dataN and calculationN. Compares the most recent valid reading before start_timestamp and
        end_timestamp and returns only instruments whose breached review status changed (including to/from no breach).
        """
    )

    def _run(
        self,
        instrument_type: str,
        instrument_subtype: Optional[str],
        db_field_name: str,
        start_timestamp: Union[str, datetime],
        end_timestamp: Union[str, datetime],
        min_e: float,
        max_e: float,
        min_n: float,
        max_n: float,
        exclude_instrument_ids: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, None, str]:
        if not instrument_type or not db_field_name:
            return "ERROR: instrument_type and db_field_name are required."

        try:
            valid_col = _validate_col(db_field_name)
        except ValueError as e:
            return f"ERROR: {e}"

        for name, ts in [("start_timestamp", start_timestamp), ("end_timestamp", end_timestamp)]:
            if isinstance(ts, str):
                try:
                    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    return "ERROR: Invalid timestamp format (use ISO)."
                if name == "start_timestamp":
                    start_timestamp = parsed
                else:
                    end_timestamp = parsed

        try:
            min_e = float(min_e)
            max_e = float(max_e)
            min_n = float(min_n)
            max_n = float(max_n)
        except Exception:
            return "ERROR: min/max easting/northing must be numeric."

        if valid_col.startswith("calculation"):
            json_path = f"$.{valid_col}"
            field_expr = (
                "CASE WHEN JSON_VALID(m.custom_fields) "
                "AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "THEN CAST(REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') AS DECIMAL(20,6)) "
                "ELSE NULL END"
            )
            reading_is_not_null = (
                "JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
            )
        else:
            field_expr = (
                f"CASE WHEN REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                f"THEN CAST(REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
            )
            reading_is_not_null = (
                f"REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$'"
            )

        exclude_clause = ""
        if exclude_instrument_ids:
            try:
                in_list = ", ".join(_quote(x) for x in exclude_instrument_ids if x)
                if in_list:
                    exclude_clause = f" AND i.instr_id NOT IN ({in_list}) "
            except Exception as e:
                return f"ERROR: Failed to process exclude_instrument_ids: {e}"

        sql = (
            "WITH eligible AS ( "
            "SELECT i.instr_id, "
            "CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) AS easting, "
            "CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) AS northing "
            "FROM instrum i JOIN location l ON i.location_id = l.id "
            "WHERE i.type1 = %(instrument_type)s "
            "AND (%(instrument_subtype)s IS NULL OR i.subtype1 = %(instrument_subtype)s) "
            "AND REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "AND REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "AND CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_e)s AND %(max_e)s "
            "AND CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_n)s AND %(max_n)s "
            + exclude_clause +
            "), start_latest AS ( "
            "SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value, "
            "ROW_NUMBER() OVER (PARTITION BY m.instr_id ORDER BY m.date1 DESC) AS rn "
            "FROM mydata m JOIN eligible e ON e.instr_id = m.instr_id "
            "WHERE m.date1 < %(start_ts)s AND " + reading_is_not_null + " ), "
            "start_chosen AS ( SELECT instr_id, reading_time, field_value FROM start_latest WHERE rn = 1 ), "
            "end_latest AS ( "
            "SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value, "
            "ROW_NUMBER() OVER (PARTITION BY m.instr_id ORDER BY m.date1 DESC) AS rn "
            "FROM mydata m JOIN eligible e ON e.instr_id = m.instr_id "
            "WHERE m.date1 < %(end_ts)s AND " + reading_is_not_null + " ), "
            "end_chosen AS ( SELECT instr_id, reading_time, field_value FROM end_latest WHERE rn = 1 ), "
            "reviews AS ( "
            "SELECT ri.instr_id, rl.review_name, "
            "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
            "riv.review_direction "
            "FROM review_instruments ri "
            "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "WHERE ri.review_field = %(review_field)s AND ri.review_status = 'ON' ), "
            "start_breaches AS ( "
            "SELECT c.instr_id, c.field_value, c.reading_time, r.review_name, r.review_value, r.review_direction, "
            "ROW_NUMBER() OVER (PARTITION BY c.instr_id ORDER BY CASE WHEN r.review_direction = 1 THEN -r.review_value ELSE r.review_value END) AS severity_rank "
            "FROM start_chosen c JOIN reviews r ON r.instr_id = c.instr_id "
            "WHERE ((r.review_direction = 1 AND c.field_value > r.review_value) OR (r.review_direction = -1 AND c.field_value < r.review_value)) ), "
            "end_breaches AS ( "
            "SELECT c.instr_id, c.field_value, c.reading_time, r.review_name, r.review_value, r.review_direction, "
            "ROW_NUMBER() OVER (PARTITION BY c.instr_id ORDER BY CASE WHEN r.review_direction = 1 THEN -r.review_value ELSE r.review_value END) AS severity_rank "
            "FROM end_chosen c JOIN reviews r ON r.instr_id = c.instr_id "
            "WHERE ((r.review_direction = 1 AND c.field_value > r.review_value) OR (r.review_direction = -1 AND c.field_value < r.review_value)) ) "
            "SELECT e.instr_id AS instrument_id, e.easting, e.northing, "
            "sb.review_name AS start_review_status, sb.review_value AS start_review_value, "
            "sc.field_value AS start_db_field_value, sc.reading_time AS start_db_field_value_timestamp, "
            "eb.review_name AS end_review_status, eb.review_value AS end_review_value, "
            "ec.field_value AS end_db_field_value, ec.reading_time AS end_db_field_value_timestamp "
            "FROM eligible e "
            "JOIN start_chosen sc ON sc.instr_id = e.instr_id "
            "JOIN end_chosen ec ON ec.instr_id = e.instr_id "
            "LEFT JOIN (SELECT instr_id, review_name, review_value FROM start_breaches WHERE severity_rank = 1) sb ON sb.instr_id = e.instr_id "
            "LEFT JOIN (SELECT instr_id, review_name, review_value FROM end_breaches WHERE severity_rank = 1) eb ON eb.instr_id = e.instr_id "
            "WHERE ((sb.review_name IS NULL AND eb.review_name IS NOT NULL) OR (sb.review_name IS NOT NULL AND eb.review_name IS NULL) OR (sb.review_name <> eb.review_name)) "
            "ORDER BY ec.reading_time DESC"
        )

        params: Dict[str, Any] = {
            "instrument_type": instrument_type,
            "instrument_subtype": instrument_subtype,
            "review_field": valid_col,
            "start_ts": start_timestamp,
            "end_ts": end_timestamp,
            "min_e": min_e,
            "max_e": max_e,
            "min_n": min_n,
            "max_n": max_n,
        }
        if valid_col.startswith("calculation"):
            params["json_path"] = json_path

        try:
            rendered = sql % {k: _quote(v) for k, v in params.items()}
        except Exception as e:
            return f"ERROR: render failed: {e}"

        try:
            logger.info("[get_map_series_review_change] SQL=%s", rendered)
            result = self.sql_tool._run(rendered)
            logger.info("[get_map_series_review_change] Raw result=%s", result)
        except Exception as e:
            return f"ERROR: unified query failed: {e}"

        if _is_no_data(result):
            return None
        try:
            rows = _parse_rows(result)
        except Exception as e:
            return f"ERROR: unified parse failed: {e}"

        if not rows:
            return None

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                e = _coerce_float(r.get("easting"))
                n = _coerce_float(r.get("northing"))
                s_fv = _coerce_float(r.get("start_db_field_value"))
                e_fv = _coerce_float(r.get("end_db_field_value"))
                s_rv = _coerce_float(r.get("start_review_value")) if r.get("start_review_value") is not None else None
                e_rv = _coerce_float(r.get("end_review_value")) if r.get("end_review_value") is not None else None
                if e is None or n is None or s_fv is None or e_fv is None:
                    raise ValueError("required numeric conversion failed")
                out.append(_MapSeriesReviewChangeRow(
                    instrument_id=r.get("instrument_id"),
                    easting=e,
                    northing=n,
                    start_review_status=r.get("start_review_status"),
                    start_review_value=s_rv,
                    start_db_field_value=s_fv,
                    start_db_field_value_timestamp=r.get("start_db_field_value_timestamp"),
                    end_review_status=r.get("end_review_status"),
                    end_review_value=e_rv,
                    end_db_field_value=e_fv,
                    end_db_field_value_timestamp=r.get("end_db_field_value_timestamp"),
                ).model_dump())
            except Exception as e:
                logger.warning("[get_map_series_review_change] Row skipped: %s row=%s", e, r)

        return pd.DataFrame(out) if out else None