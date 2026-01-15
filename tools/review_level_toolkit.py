import logging
import re
import json
import datetime as dt
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Type, Tuple, Literal
import pandas as pd

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import BaseTool
from tools.sql_security_toolkit import GeneralSQLQueryTool

logger = logging.getLogger(__name__)

_VALID_DB_FIELD_RE = re.compile(r"^(data|calculation)\d+$")

NO_DATA_MSG = "No data was found in the database matching the specified search criteria."

def _parse_any_datetime(dt_str: str) -> datetime:
    """Parse datetime from ISO or verbose string, handling 00:00:00 AM edge case."""
    if "00:00:00 AM" in dt_str:
        dt_str = dt_str.replace("00:00:00 AM", "12:00:00 AM")
        
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Verbose formats (matching create_output_toolkit)
    # Note: %#d is Windows-specific, %d is standard
    for fmt in ("%d %B %Y %I:%M:%S %p", "%#d %B %Y %#I:%M:%S %p"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
            
    raise ValueError("Invalid datetime format")

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
    mysql_version_major: int = Field(8, description="MySQL major version used to select legacy-safe SQL paths.", exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _use_legacy_sql(self) -> bool:
        try:
            return int(self.mysql_version_major) < 8
        except Exception:
            return False

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


class _ReviewChangeAcrossPeriod(BaseModel):
    """Schema describing a change in review status between two timestamps."""
    instrument_id: str = Field(..., description="Instrument ID (from instrum.instr_id).")
    db_field_name: str = Field(..., description="System field name (dataN or calculationN).")
    start_review_name: Optional[str] = Field(None, description="Review level breached just before start_timestamp; DataFrame output shows NaN when unbreached.")
    start_review_value: Optional[float] = Field(None, description="Threshold value for the breached start review level (if any).")
    start_field_value: float = Field(..., description="Reading value immediately before start_timestamp within buffer.")
    start_field_value_timestamp: datetime = Field(..., description="Timestamp of the start reading.")
    end_review_name: Optional[str] = Field(None, description="Review level breached just before end_timestamp; DataFrame output shows NaN when unbreached.")
    end_review_value: Optional[float] = Field(None, description="Threshold value for the breached end review level (if any).")
    end_field_value: float = Field(..., description="Reading value immediately before end_timestamp within buffer.")
    end_field_value_timestamp: datetime = Field(..., description="Timestamp of the end reading.")

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


class _ReviewChangesAcrossPeriodInput(BaseModel):
    instrument_type: Optional[str] = Field(None, description="Optional instrument type filter (instrum.type1).")
    instrument_subtype: Optional[str] = Field(None, description="Optional instrument subtype filter (instrum.subtype1).")
    db_field_name: Optional[str] = Field(None, description="Optional data/calculation field to restrict review discovery.")
    start_timestamp: Union[str, datetime] = Field(..., description="Period start boundary (ISO string or datetime).")
    end_timestamp: Union[str, datetime] = Field(..., description="Period end boundary (ISO string or datetime).")
    start_buffer: Optional[Union[int, float]] = Field(
        None,
        description="Maximum days before start_timestamp allowed for the start reading.",
    )
    end_buffer: Optional[Union[int, float]] = Field(
        None,
        description="Maximum days before end_timestamp allowed for the end reading.",
    )
    change_direction: Literal["up", "down", "both"] = Field(
        "up",
        description="Return only more-severe ('up'), less-severe ('down'), or all ('both') review changes.",
    )


class GetReviewStatusByValueTool(BaseTool, _BaseQueryTool):
    """
    Returns the review level status (most severe breached review level name) for one or more database field values at specified instruments and fields. Non-breaching outputs appear with NaN review_name in the DataFrame; returns None only when no active reviews are found.

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
        Returns the review level status (most severe breached review level name) for one or more database field values at specified instruments and fields. Non-breaching outputs appear with NaN review_name in the DataFrame; returns None only when no active reviews are found.

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
                params_inline = "( " + " UNION ALL ".join(values_sql_parts) + " ) AS p"
            except Exception as e:
                return f"ERROR: render parameters failed: {e}"
            sql = (
                "SELECT p.instr_id AS instrument_id, p.review_field AS db_field_name, p.field_value AS db_field_value, "
                " (SELECT rl.review_name "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  JOIN review_levels rl ON riv.review_level_id = rl.id "
                "  WHERE ri.instr_id = p.instr_id AND ri.review_field = p.review_field AND ri.review_status = 'ON' "
                "  AND ((riv.review_direction = 1 AND p.field_value > CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))) "
                "       OR (riv.review_direction = -1 AND p.field_value < CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)))) "
                "  ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END "
                "  LIMIT 1) AS review_name "
                "FROM " + params_inline + " "
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
                timestamp = _parse_any_datetime(timestamp)
            except Exception:
                return "ERROR: Invalid timestamp format (use ISO or 'D Month YYYY H:MM:SS AM/PM')."

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
                params_inline = "( " + " UNION ALL ".join(values_sql_parts) + " ) AS p"
            except Exception as e:
                return f"ERROR: render parameters failed: {e}"

            if field.startswith("calculation"):
                json_path = f"$.$FIELD$"
                json_path = json_path.replace("$FIELD$", field)
                field_expr = (
                    "CASE WHEN JSON_VALID(m1.custom_fields) "
                    f"AND JSON_EXTRACT(m1.custom_fields, {_quote(json_path)}) IS NOT NULL "
                    f"AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, {_quote(json_path)}))), '') IS NOT NULL "
                    f"THEN CAST(REPLACE(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, {_quote(json_path)})), ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"JSON_VALID(m1.custom_fields) AND JSON_EXTRACT(m1.custom_fields, {_quote(json_path)}) IS NOT NULL "
                    f"AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, {_quote(json_path)}))), '') IS NOT NULL"
                )
                valid_col = field
            else:
                try:
                    valid_col = _validate_col(field)
                except ValueError as e:
                    return f"ERROR: {e}"
                field_expr = (
                    f"CASE WHEN NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL "
                    f"THEN CAST(REPLACE(m1.{valid_col}, ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL"
                )

            reading_is_not_null_sub = reading_is_not_null.replace("m1.", "m2.")
            reading_is_not_null_legacy = reading_is_not_null

            review_value_expr = "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))"
            sql = (
                "SELECT lr.instr_id AS instrument_id, " + _quote(valid_col) + " AS db_field_name, "
                " lr.field_value AS db_field_value, lr.reading_time AS db_field_value_timestamp, "
                " (SELECT rl.review_name "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  JOIN review_levels rl ON riv.review_level_id = rl.id "
                f"  WHERE ri.instr_id = lr.instr_id AND ri.review_field = {_quote(valid_col)} AND ri.review_status = 'ON' "
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                "  ORDER BY CASE WHEN riv.review_direction = 1 THEN -" + review_value_expr + " ELSE " + review_value_expr + " END "
                "  LIMIT 1) AS review_name "
                "FROM ( "
                "  SELECT m1.instr_id, m1.date1 AS reading_time, " + field_expr + " AS field_value "
                "  FROM mydata m1 JOIN " + params_inline + " ON p.instr_id = m1.instr_id "
                f"  WHERE m1.date1 < {_quote(timestamp)} AND " + reading_is_not_null_legacy + " "
                "    AND NOT EXISTS ("
                "      SELECT 1 FROM mydata m2 "
                "      WHERE m2.instr_id = m1.instr_id AND m2.date1 < " + _quote(timestamp) + " AND " + reading_is_not_null_sub + " "
                "        AND (m2.date1 > m1.date1 OR (m2.date1 = m1.date1 AND m2.id > m1.id))"
                "    )"
                " ) lr "
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
            params_inline = "( " + " UNION ALL ".join(values_sql_parts) + " ) AS p"

            legacy_sql = (
                "SELECT ri.instr_id AS instrument_id, " + _quote(field) + " AS db_field_name, rl.review_name, "
                " CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
                " CASE WHEN riv.review_direction = 1 THEN 'upper' ELSE 'lower' END AS review_direction, "
                " CONCAT('#', aci.aaa_color) AS review_color "
                " FROM " + params_inline +
                " JOIN review_instruments ri ON ri.instr_id = p.instr_id AND ri.review_status = 'ON' AND ri.review_field = " + _quote(field) +
                " JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                " JOIN review_levels rl ON riv.review_level_id = rl.id "
                " JOIN aaa_color_info aci ON rl.id = aci.review_id "
                " ORDER BY aci.`order`"
            )

            sql = legacy_sql

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
            params_inline = "( " + " UNION ALL ".join(values_sql_parts) + " ) AS p"

            modern_sql = (
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

            legacy_sql = (
                " SELECT p.instr_id AS instrument_id, " + _quote(field) + " AS db_field_name, p.review_name, "
                " CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value, "
                " CASE WHEN riv.review_direction = 1 THEN 'upper' ELSE 'lower' END AS review_direction "
                " FROM " + params_inline +
                " JOIN review_instruments ri ON ri.instr_id = p.instr_id AND ri.review_field = " + _quote(field) + " AND ri.review_status = 'ON' "
                " JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                " JOIN review_levels rl ON riv.review_level_id = rl.id AND rl.review_name = p.review_name "
                " WHERE (ri.effective_from IS NULL OR ri.effective_from <= NOW()) "
            )

            sql = legacy_sql if self._use_legacy_sql else modern_sql

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
                timestamp = _parse_any_datetime(timestamp)
            except Exception:
                return "ERROR: Invalid timestamp format. Use ISO format or 'D Month YYYY H:MM:SS AM/PM'."

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
                    "CASE WHEN JSON_VALID(m1.custom_fields) "
                    "AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL "
                    "THEN CAST(REPLACE(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s)), ',', '') AS DECIMAL(20,6)) "
                    "ELSE NULL END"
                )
                reading_is_not_null = (
                    "JSON_VALID(m1.custom_fields) AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL"
                )
            else:
                field_expr = (
                    f"CASE WHEN NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL "
                    f"THEN CAST(REPLACE(m1.{valid_col}, ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL"
                )

            severity_order = (
                "CASE WHEN riv.review_direction = 1 "
                "THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) "
                "ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END"
            )
            review_value_expr = "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))"

            filters_join = ""
            filters_where: List[str] = []
            if instrument_type or instrument_subtype:
                filters_join = "JOIN instrum i ON lr.instr_id = i.instr_id "
                if instrument_type:
                    filters_where.append("i.type1 = %(instrument_type)s")
                if instrument_subtype:
                    filters_where.append("i.subtype1 = %(instrument_subtype)s")
            filters_where_clause = ("WHERE " + " AND ".join(filters_where)) if filters_where else ""

            review_name_filter = " AND rl.review_name = %(review_name)s" if review_name else ""

            reading_clause_m2 = reading_is_not_null.replace("m1.", "m2.")

            sql = (
                "SELECT lr.instr_id AS instrument_id, %(review_field)s AS db_field_name, "
                " (SELECT rl.review_name "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  JOIN review_levels rl ON riv.review_level_id = rl.id "
                "  WHERE ri.instr_id = lr.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                + review_name_filter +
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                "  ORDER BY " + severity_order + " "
                "  LIMIT 1) AS review_name, "
                " (SELECT " + review_value_expr + " "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  WHERE ri.instr_id = lr.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                + review_name_filter +
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                "  ORDER BY " + severity_order + " "
                "  LIMIT 1) AS review_value, "
                " lr.field_value, lr.reading_time AS field_value_timestamp "
                "FROM ( "
                "  SELECT m1.instr_id, m1.date1 AS reading_time, " + field_expr + " AS field_value "
                "  FROM mydata m1 "
                "  JOIN review_instruments ri ON m1.instr_id = ri.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                "  WHERE m1.date1 < %(ts)s AND " + reading_is_not_null + " "
                "    AND NOT EXISTS ( "
                "      SELECT 1 FROM mydata m2 "
                "      WHERE m2.instr_id = m1.instr_id AND " + reading_clause_m2 + " AND m2.date1 < %(ts)s "
                "        AND (m2.date1 > m1.date1 OR (m2.date1 = m1.date1 AND m2.id > m1.id)) "
                "    ) "
                ") lr "
                + filters_join + " " + filters_where_clause + " "
                "WHERE EXISTS ( "
                "  SELECT 1 "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  WHERE ri.instr_id = lr.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                + review_name_filter +
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                ") "
                "ORDER BY lr.reading_time DESC"
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
                    "CASE WHEN JSON_VALID(m1.custom_fields) "
                    "AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL "
                    "THEN CAST(REPLACE(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s)), ',', '') AS DECIMAL(20,6)) "
                    "ELSE NULL END"
                )
                reading_is_not_null = (
                    "JSON_VALID(m1.custom_fields) AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL"
                )
            else:
                field_expr = (
                    f"CASE WHEN NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL "
                    f"THEN CAST(REPLACE(m1.{valid_col}, ',', '') AS DECIMAL(20,6)) ELSE NULL END"
                )
                reading_is_not_null = (
                    f"NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL"
                )

            severity_order = (
                "CASE WHEN riv.review_direction = 1 "
                "THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) "
                "ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END"
            )
            review_value_expr = "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))"

            filters_join = ""
            filters_where: List[str] = []
            if instrument_type or instrument_subtype:
                filters_join = "JOIN instrum i ON lr.instr_id = i.instr_id "
                if instrument_type:
                    filters_where.append("i.type1 = %(instrument_type)s")
                if instrument_subtype:
                    filters_where.append("i.subtype1 = %(instrument_subtype)s")
            filters_where_clause = ("WHERE " + " AND ".join(filters_where)) if filters_where else ""

            review_name_filter = " AND rl.review_name = %(review_name)s" if review_name else ""

            reading_clause_m2 = reading_is_not_null.replace("m1.", "m2.")

            sql = (
                "SELECT lr.instr_id AS instrument_id, %(review_field)s AS db_field_name, "
                " (SELECT rl.review_name "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  JOIN review_levels rl ON riv.review_level_id = rl.id "
                "  WHERE ri.instr_id = lr.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                + review_name_filter +
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                "  ORDER BY " + severity_order + " "
                "  LIMIT 1) AS review_name, "
                " (SELECT " + review_value_expr + " "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  WHERE ri.instr_id = lr.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                + review_name_filter +
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                "  ORDER BY " + severity_order + " "
                "  LIMIT 1) AS review_value, "
                " lr.field_value, lr.reading_time AS field_value_timestamp "
                "FROM ( "
                "  SELECT m1.instr_id, m1.date1 AS reading_time, " + field_expr + " AS field_value "
                "  FROM mydata m1 "
                "  JOIN review_instruments ri ON m1.instr_id = ri.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                "  WHERE m1.date1 < %(ts)s AND " + reading_is_not_null + " "
                "    AND NOT EXISTS ( "
                "      SELECT 1 FROM mydata m2 "
                "      WHERE m2.instr_id = m1.instr_id AND " + reading_clause_m2 + " AND m2.date1 < %(ts)s "
                "        AND (m2.date1 > m1.date1 OR (m2.date1 = m1.date1 AND m2.id > m1.id)) "
                "    ) "
                ") lr "
                + filters_join + " " + filters_where_clause + " "
                "WHERE EXISTS ( "
                "  SELECT 1 "
                "  FROM review_instruments ri "
                "  JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "  WHERE ri.instr_id = lr.instr_id AND ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
                + review_name_filter +
                "    AND ((riv.review_direction = 1 AND lr.field_value > " + review_value_expr + ") "
                "         OR (riv.review_direction = -1 AND lr.field_value < " + review_value_expr + ")) "
                ") "
                "ORDER BY lr.reading_time DESC"
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


class GetReviewChangesAcrossPeriodTool(BaseTool, _BaseQueryTool):
    """
    Finds changes in review level at instruments across a specified time period between `start_timestamp` and `end_timestamp`.
    Optional inputs:
    - instrument type (`instrument_type`), subtype (`instrument_subtype`) and data field name (`db_field_name`)
    - change direction (`change_direction`)
    - buffer windows  (`start_buffer`, `end_buffer`) before the period start and end within which to look for the most recent readings

    Returns a pandas DataFrame with one row per breached instrument (columns match ReviewChangeAcrossPeriod), or None, or an ERROR. 
    """

    name: str = "get_review_changes_across_period_tool"
    description: str = (
        """
        Detects review-level status changes between two timestamps for instruments filtered by type/subtype and optional db_field_name.
        Takes `start_timestamp`, `end_timestamp`, optional buffer durations in days (`start_buffer`, `end_buffer`) and a `change_direction` flag ('up', 'down', 'both'). Returns only combinations where the breached review status differs between the two readings (including transitions to/from unbreached) provided both readings exist within their respective buffers.
        """
    )

    args_schema: Type[BaseModel] = _ReviewChangesAcrossPeriodInput

    def _run(
        self,
        instrument_type: Optional[str],
        instrument_subtype: Optional[str],
        db_field_name: Optional[str],
        start_timestamp: Union[str, datetime],
        end_timestamp: Union[str, datetime],
        start_buffer: Optional[Union[int, float]] = None,
        end_buffer: Optional[Union[int, float]] = None,
        change_direction: str = "up",
    ) -> Union[pd.DataFrame, None, str]:
        def _parse_timestamp(val: Union[str, datetime], label: str) -> Union[datetime, str]:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return _parse_any_datetime(val)
                except Exception:
                    return f"ERROR: Invalid {label} format (use ISO or 'D Month YYYY H:MM:SS AM/PM')."
            return f"ERROR: {label} must be a datetime or ISO 8601 string."

        MAX_BUFFER_SECONDS = 7 * 86400.0

        def _coerced_buffer_seconds(val: Optional[Union[int, float]]) -> Optional[float]:
            """Best-effort buffer parsing in seconds; returns None when missing/invalid."""
            if val is None:
                return None
            try:
                days = float(val)
            except Exception:
                return None
            if days < 0:
                return None
            seconds = days * 86400.0
            return min(seconds, MAX_BUFFER_SECONDS)

        start_ts = _parse_timestamp(start_timestamp, "start_timestamp")
        if isinstance(start_ts, str):
            return start_ts
        end_ts = _parse_timestamp(end_timestamp, "end_timestamp")
        if isinstance(end_ts, str):
            return end_ts
        if end_ts <= start_ts:
            return "ERROR: end_timestamp must be after start_timestamp."

        total_seconds = (end_ts - start_ts).total_seconds()
        if total_seconds <= 0:
            return "ERROR: start_timestamp and end_timestamp must be distinct with end > start."

        default_buffer = min(total_seconds / 2.0, MAX_BUFFER_SECONDS)

        parsed_start_buffer = _coerced_buffer_seconds(start_buffer)
        if parsed_start_buffer is None or parsed_start_buffer <= 0:
            parsed_start_buffer = default_buffer

        parsed_end_buffer = _coerced_buffer_seconds(end_buffer)
        if parsed_end_buffer is None or parsed_end_buffer <= 0:
            parsed_end_buffer = default_buffer
        parsed_end_buffer = min(parsed_end_buffer, total_seconds, MAX_BUFFER_SECONDS)

        start_lower_bound = start_ts - timedelta(seconds=parsed_start_buffer) if parsed_start_buffer is not None else None
        end_lower_bound = end_ts - timedelta(seconds=parsed_end_buffer)

        direction = (change_direction or "up").lower()
        if direction not in {"up", "down", "both"}:
            return "ERROR: change_direction must be 'up', 'down', or 'both'."

        instrument_type = instrument_type or None
        instrument_subtype = instrument_subtype or None

        if db_field_name:
            try:
                fields = [_validate_col(db_field_name)]
            except ValueError as e:
                return f"ERROR: {e}"
        else:
            join_clause = ""
            where_parts = ["ri.review_status = 'ON'"]
            if instrument_type or instrument_subtype:
                join_clause = "JOIN instrum i ON ri.instr_id = i.instr_id "
                if instrument_type:
                    where_parts.append("i.type1 = %(instrument_type)s")
                if instrument_subtype:
                    where_parts.append("i.subtype1 = %(instrument_subtype)s")
            discover_sql = (
                "SELECT DISTINCT ri.review_field AS db_field_name "
                "FROM review_instruments ri "
                + join_clause +
                " WHERE " + " AND ".join(where_parts)
            )
            try:
                rendered_discover = discover_sql % {
                    "instrument_type": _quote(instrument_type),
                    "instrument_subtype": _quote(instrument_subtype),
                }
            except Exception as e:
                return f"ERROR: render discover failed: {e}"
            try:
                logger.info("[get_review_changes/discover] SQL=%s", rendered_discover)
                discover_result = self.sql_tool._run(rendered_discover)
                logger.info("[get_review_changes/discover] Raw result=%s", discover_result)
            except Exception as e:
                return f"ERROR: discover query failed: {e}"
            if _is_no_data(discover_result):
                return None
            try:
                field_rows = _parse_rows(discover_result)
            except Exception as e:
                return f"ERROR: discover parse failed: {e}"
            fields = []
            for row in field_rows:
                f = row.get("db_field_name")
                if not f:
                    continue
                try:
                    fields.append(_validate_col(str(f)))
                except ValueError:
                    continue
            fields = list(dict.fromkeys(fields))
            if not fields:
                return None

        def _field_sql_parts(valid_col: str) -> Tuple[str, str, Dict[str, Any]]:
            if valid_col.startswith("calculation"):
                json_path = f"$.{valid_col}"
                field_expr_local = (
                    "CASE WHEN JSON_VALID(m1.custom_fields) "
                    "AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL "
                    "THEN CAST(REPLACE(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s)), ',', '') AS DECIMAL(20,6)) "
                    "ELSE NULL END"
                )
                reading_clause = (
                    "JSON_VALID(m1.custom_fields) AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                    "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL"
                )
                return field_expr_local, reading_clause, {"json_path": json_path}
            field_expr_local = (
                f"CASE WHEN NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL "
                f"THEN CAST(REPLACE(m1.{valid_col}, ',', '') AS DECIMAL(20,6)) ELSE NULL END"
            )
            reading_clause = (
                f"NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL"
            )
            return field_expr_local, reading_clause, {}

        active_instr_join = ""
        active_instr_filters = [
            "ri.review_status = 'ON'",
            "ri.review_field = %(review_field)s",
        ]
        if instrument_type or instrument_subtype:
            active_instr_join = "JOIN instrum i ON ri.instr_id = i.instr_id "
            if instrument_type:
                active_instr_filters.append("i.type1 = %(instrument_type)s")
            if instrument_subtype:
                active_instr_filters.append("i.subtype1 = %(instrument_subtype)s")
        active_instr_where = " AND ".join(active_instr_filters)

        where_clause = "WHERE NOT (sr.review_name <=> er.review_name)"
        if instrument_type or instrument_subtype:
            exists_conditions: List[str] = []
            if instrument_type:
                exists_conditions.append("i.type1 = %(instrument_type)s")
            if instrument_subtype:
                exists_conditions.append("i.subtype1 = %(instrument_subtype)s")
            where_clause += (
                " AND EXISTS (SELECT 1 FROM instrum i WHERE i.instr_id = se.instr_id "
                "AND " + " AND ".join(exists_conditions) + ")"
            )

        aggregated: List[Dict[str, Any]] = []
        for valid_col in fields:
            field_expr, reading_is_not_null, extra_params = _field_sql_parts(valid_col)

            start_buffer_clause = " AND m1.date1 >= %(start_lower)s" if start_lower_bound is not None else ""

            active_instr_sql = (
                "SELECT DISTINCT ri.instr_id FROM review_instruments ri "
                + active_instr_join +
                "WHERE " + active_instr_where
            )

            review_value_expr = (
                "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))"
            )
            severity_sort_expr = (
                "CASE WHEN riv.review_direction = 1 THEN -" + review_value_expr + " ELSE " + review_value_expr + " END"
            )

            reading_clause_m2 = reading_is_not_null.replace("m1.", "m2.")

            start_latest_sql = (
                "SELECT ai.instr_id, m1.date1 AS start_field_value_timestamp, "
                + field_expr + " AS start_field_value "
                "FROM mydata m1 "
                "JOIN (" + active_instr_sql + ") ai ON ai.instr_id = m1.instr_id "
                "WHERE " + reading_is_not_null +
                " AND m1.date1 < %(start_ts)s" + start_buffer_clause +
                " AND m1.id = ("
                "SELECT m2.id FROM mydata m2 WHERE m2.instr_id = m1.instr_id AND "
                + reading_clause_m2 + " AND m2.date1 < %(start_ts)s" + start_buffer_clause +
                " ORDER BY m2.date1 DESC, m2.id DESC LIMIT 1)"
            )

            end_latest_sql = (
                "SELECT ai.instr_id, m1.date1 AS end_field_value_timestamp, "
                + field_expr + " AS end_field_value "
                "FROM mydata m1 "
                "JOIN (" + active_instr_sql + ") ai ON ai.instr_id = m1.instr_id "
                "WHERE " + reading_is_not_null +
                " AND m1.date1 < %(end_ts)s AND m1.date1 >= %(end_lower)s"
                " AND m1.id = ("
                "SELECT m2.id FROM mydata m2 WHERE m2.instr_id = m1.instr_id AND "
                + reading_clause_m2 + " AND m2.date1 < %(end_ts)s AND m2.date1 >= %(end_lower)s"
                " ORDER BY m2.date1 DESC, m2.id DESC LIMIT 1)"
            )

            base_reviews_sql = (
                "SELECT ri.instr_id, rl.review_name, "
                + review_value_expr + " AS review_value_clean, "
                "riv.review_direction, "
                + severity_sort_expr + " AS severity_sort "
                "FROM review_instruments ri "
                "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "JOIN review_levels rl ON riv.review_level_id = rl.id "
                "WHERE ri.review_status = 'ON' AND ri.review_field = %(review_field)s "
            )

            start_min_severity = (
                "SELECT MIN(b2.severity_sort) FROM (" + base_reviews_sql + ") b2 "
                "WHERE b2.instr_id = se.instr_id "
                " AND ((b2.review_direction = 1 AND se.start_field_value > b2.review_value_clean) "
                "OR (b2.review_direction = -1 AND se.start_field_value < b2.review_value_clean))"
            )

            end_min_severity = (
                "SELECT MIN(b2.severity_sort) FROM (" + base_reviews_sql + ") b2 "
                "WHERE b2.instr_id = se.instr_id "
                " AND ((b2.review_direction = 1 AND ee.end_field_value > b2.review_value_clean) "
                "OR (b2.review_direction = -1 AND ee.end_field_value < b2.review_value_clean))"
            )

            sql = (
                "SELECT se.instr_id AS instrument_id, %(review_field)s AS db_field_name, "
                "sr.review_name AS start_review_name, sr.review_value_clean AS start_review_value, "
                "se.start_field_value, se.start_field_value_timestamp, "
                "er.review_name AS end_review_name, er.review_value_clean AS end_review_value, "
                "ee.end_field_value, ee.end_field_value_timestamp, "
                "sr.severity_sort AS start_severity_sort, er.severity_sort AS end_severity_sort "
                "FROM (" + start_latest_sql + ") se "
                "JOIN (" + end_latest_sql + ") ee ON ee.instr_id = se.instr_id "
                "LEFT JOIN (" + base_reviews_sql + ") sr ON sr.instr_id = se.instr_id "
                " AND ((sr.review_direction = 1 AND se.start_field_value > sr.review_value_clean) OR (sr.review_direction = -1 AND se.start_field_value < sr.review_value_clean)) "
                " AND sr.severity_sort = (" + start_min_severity + ") "
                "LEFT JOIN (" + base_reviews_sql + ") er ON er.instr_id = se.instr_id "
                " AND ((er.review_direction = 1 AND ee.end_field_value > er.review_value_clean) OR (er.review_direction = -1 AND ee.end_field_value < er.review_value_clean)) "
                " AND er.severity_sort = (" + end_min_severity + ") "
                + where_clause +
                " ORDER BY ee.end_field_value_timestamp DESC"
            )

            params: Dict[str, Any] = {
                "instrument_type": instrument_type,
                "instrument_subtype": instrument_subtype,
                "review_field": valid_col,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "end_lower": end_lower_bound,
            }
            if start_lower_bound is not None:
                params["start_lower"] = start_lower_bound
            params.update(extra_params)

            try:
                rendered = sql % {k: _quote(v) for k, v in params.items()}
            except Exception as e:
                logger.warning("[get_review_changes/%s] Render failed: %s", valid_col, e)
                continue
            try:
                logger.info("[get_review_changes/%s] SQL=%s", valid_col, rendered)
                result = self.sql_tool._run(rendered)
                logger.info("[get_review_changes/%s] Raw result=%s", valid_col, result)
            except Exception as e:
                logger.warning("[get_review_changes/%s] Query failed: %s", valid_col, e)
                continue
            if _is_no_data(result):
                continue
            try:
                rows = _parse_rows(result)
            except Exception as e:
                logger.warning("[get_review_changes/%s] Parse failed: %s", valid_col, e)
                continue

            for r in rows:
                try:
                    start_val = _coerce_float(r.get("start_field_value"))
                    end_val = _coerce_float(r.get("end_field_value"))
                    if start_val is None or end_val is None:
                        raise ValueError("missing numeric readings")
                    start_review_value = _coerce_float(r.get("start_review_value"))
                    end_review_value = _coerce_float(r.get("end_review_value"))
                    start_severity_sort = _coerce_float(r.get("start_severity_sort"))
                    end_severity_sort = _coerce_float(r.get("end_severity_sort"))

                    def _severity_score(val: Optional[float]) -> float:
                        return float(val) if val is not None else 10**9

                    start_score = _severity_score(start_severity_sort)
                    end_score = _severity_score(end_severity_sort)

                    changed = (
                        (r.get("start_review_name") != r.get("end_review_name"))
                        or (start_score != end_score)
                        or (start_review_value != end_review_value)
                    )
                    if not changed:
                        continue

                    include = False
                    if direction == "both":
                        include = True
                    elif direction == "up" and end_score < start_score:
                        include = True
                    elif direction == "down" and end_score > start_score:
                        include = True
                    if not include:
                        continue

                    aggregated.append(_ReviewChangeAcrossPeriod(
                        instrument_id=r.get("instrument_id"),
                        db_field_name=valid_col,
                        start_review_name=r.get("start_review_name"),
                        start_review_value=start_review_value,
                        start_field_value=start_val,
                        start_field_value_timestamp=r.get("start_field_value_timestamp"),
                        end_review_name=r.get("end_review_name"),
                        end_review_value=end_review_value,
                        end_field_value=end_val,
                        end_field_value_timestamp=r.get("end_field_value_timestamp"),
                    ).model_dump())
                except Exception as e:
                    logger.warning("[get_review_changes/%s] Row skipped: %s row=%s", valid_col, e, r)

        logger.info(
            "[get_review_changes] Aggregated %d review changes across %d fields",
            len(aggregated),
            len(fields),
        )
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
                timestamp = _parse_any_datetime(timestamp)
            except Exception:
                return "ERROR: Invalid timestamp format (use ISO or 'D Month YYYY H:MM:SS AM/PM')."

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
                "CASE WHEN JSON_VALID(m1.custom_fields) "
                "AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL "
                "THEN CAST(REPLACE(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s)), ',', '') AS DECIMAL(20,6)) "
                "ELSE NULL END"
            )
            reading_is_not_null = (
                "JSON_VALID(m1.custom_fields) AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL"
            )
        else:
            field_expr = (
                f"CASE WHEN NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL "
                f"THEN CAST(REPLACE(m1.{valid_col}, ',', '') AS DECIMAL(20,6)) ELSE NULL END"
            )
            reading_is_not_null = (
                f"NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL"
            )

        exclude_clause = ""
        if exclude_instrument_ids:
            try:
                in_list = ", ".join(_quote(x) for x in exclude_instrument_ids if x)
                if in_list:
                    exclude_clause = f" AND i.instr_id NOT IN ({in_list}) "
            except Exception as e:
                return f"ERROR: Failed to process exclude_instrument_ids: {e}"

        reading_clause_m2 = reading_is_not_null.replace("m1.", "m2.")

        eligible_sql = (
            "SELECT i.instr_id, "
            "CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) AS easting, "
            "CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) AS northing "
            "FROM instrum i JOIN location l ON i.location_id = l.id "
            "WHERE i.type1 = %(instrument_type)s "
            "AND (%(instrument_subtype)s IS NULL OR i.subtype1 = %(instrument_subtype)s) "
            "AND CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_e)s AND %(max_e)s "
            "AND CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_n)s AND %(max_n)s "
            + exclude_clause
        )

        latest_reading_sql = (
            "SELECT e.instr_id, m1.date1 AS reading_time, " + field_expr + " AS field_value "
            "FROM mydata m1 JOIN (" + eligible_sql + ") AS e ON e.instr_id = m1.instr_id "
            "WHERE " + reading_is_not_null + " AND m1.date1 < %(ts)s "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM mydata m2 WHERE m2.instr_id = m1.instr_id AND " + reading_clause_m2 + " AND m2.date1 < %(ts)s "
            "    AND (m2.date1 > m1.date1 OR (m2.date1 = m1.date1 AND m2.id > m1.id))"
            ")"
        )

        review_value_expr = (
            "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))"
        )
        severity_sort_expr = (
            "CASE WHEN riv.review_direction = 1 THEN -" + review_value_expr + " ELSE " + review_value_expr + " END"
        )

        active_reviews_base = (
            "SELECT ri.instr_id, rl.review_name, " + review_value_expr + " AS review_value, riv.review_direction, " + severity_sort_expr + " AS severity_sort "
            "FROM review_instruments ri "
            "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "WHERE ri.review_field = %(review_field)s AND ri.review_status = 'ON'"
        )

        active_reviews_ranked = (
            "SELECT ar_base.instr_id, ar_base.review_name, ar_base.review_value, ar_base.review_direction, ar_base.severity_sort, "
            "(SELECT COUNT(*) + 1 FROM (" + active_reviews_base + ") AS ar2 WHERE ar2.instr_id = ar_base.instr_id AND ar2.severity_sort < ar_base.severity_sort) AS severity_rank "
            "FROM (" + active_reviews_base + ") AS ar_base"
        )

        sql = (
            "SELECT e.instr_id AS instrument_id, e.easting, e.northing, "
            "(SELECT CONCAT_WS('|', ar1.review_name, ar1.review_value) FROM (" + active_reviews_ranked + ") AS ar1 WHERE ar1.instr_id = e.instr_id "
            " AND ((ar1.review_direction = 1 AND lr.field_value > ar1.review_value) OR (ar1.review_direction = -1 AND lr.field_value < ar1.review_value)) "
            " ORDER BY ar1.severity_rank LIMIT 1) AS review_pair, "
            "lr.field_value AS db_field_value, lr.reading_time AS db_field_value_timestamp "
            "FROM (" + eligible_sql + ") AS e "
            "JOIN (" + latest_reading_sql + ") AS lr ON lr.instr_id = e.instr_id "
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
                pair_raw = r.get("review_pair")
                rv_status: Optional[str] = None
                rv_value: Optional[float] = None
                if pair_raw is not None:
                    try:
                        status_part, value_part = str(pair_raw).split("|", 1)
                        rv_status = status_part or None
                        rv_value = _coerce_float(value_part) if value_part != "" else None
                    except ValueError:
                        rv_status = str(pair_raw) or None
                if e is None or n is None or fv is None:
                    raise ValueError("required numeric conversion failed")
                out.append(_MapSeriesReviewStatusRow(
                    instrument_id=r.get("instrument_id"),
                    easting=e,
                    northing=n,
                    review_status=rv_status,
                    review_value=rv_value,
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
                    parsed = _parse_any_datetime(ts)
                except Exception:
                    return "ERROR: Invalid timestamp format (use ISO or 'D Month YYYY H:MM:SS AM/PM')."
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
                "CASE WHEN JSON_VALID(m1.custom_fields) "
                "AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL "
                "THEN CAST(REPLACE(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s)), ',', '') AS DECIMAL(20,6)) "
                "ELSE NULL END"
            )
            reading_is_not_null = (
                "JSON_VALID(m1.custom_fields) AND JSON_EXTRACT(m1.custom_fields, %(json_path)s) IS NOT NULL "
                "AND NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m1.custom_fields, %(json_path)s))), '') IS NOT NULL"
            )
        else:
            field_expr = (
                f"CASE WHEN NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL "
                f"THEN CAST(REPLACE(m1.{valid_col}, ',', '') AS DECIMAL(20,6)) ELSE NULL END"
            )
            reading_is_not_null = (
                f"NULLIF(TRIM(REPLACE(m1.{valid_col}, ',', '')), '') IS NOT NULL"
            )

        exclude_clause = ""
        if exclude_instrument_ids:
            try:
                in_list = ", ".join(_quote(x) for x in exclude_instrument_ids if x)
                if in_list:
                    exclude_clause = f" AND i.instr_id NOT IN ({in_list}) "
            except Exception as e:
                return f"ERROR: Failed to process exclude_instrument_ids: {e}"

        reading_clause_m2 = reading_is_not_null.replace("m1.", "m2.")

        eligible_sql = (
            "SELECT i.instr_id, "
            "CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) AS easting, "
            "CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) AS northing "
            "FROM instrum i JOIN location l ON i.location_id = l.id "
            "WHERE i.type1 = %(instrument_type)s "
            "AND (%(instrument_subtype)s IS NULL OR i.subtype1 = %(instrument_subtype)s) "
            "AND CAST(REPLACE(NULLIF(TRIM(l.easting), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_e)s AND %(max_e)s "
            "AND CAST(REPLACE(NULLIF(TRIM(l.northing), ''), ',', '') AS DECIMAL(20,6)) BETWEEN %(min_n)s AND %(max_n)s "
            + exclude_clause
        )

        start_latest_sql = (
            "SELECT e.instr_id, m1.date1 AS start_field_value_timestamp, " + field_expr + " AS start_field_value "
            "FROM mydata m1 JOIN (" + eligible_sql + ") AS e ON e.instr_id = m1.instr_id "
            "WHERE " + reading_is_not_null + " AND m1.date1 < %(start_ts)s "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM mydata m2 WHERE m2.instr_id = m1.instr_id AND " + reading_clause_m2 + " AND m2.date1 < %(start_ts)s "
            "    AND (m2.date1 > m1.date1 OR (m2.date1 = m1.date1 AND m2.id > m1.id))"
            ")"
        )

        end_latest_sql = (
            "SELECT e.instr_id, m1.date1 AS end_field_value_timestamp, " + field_expr + " AS end_field_value "
            "FROM mydata m1 JOIN (" + eligible_sql + ") AS e ON e.instr_id = m1.instr_id "
            "WHERE " + reading_is_not_null + " AND m1.date1 < %(end_ts)s "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM mydata m2 WHERE m2.instr_id = m1.instr_id AND " + reading_clause_m2 + " AND m2.date1 < %(end_ts)s "
            "    AND (m2.date1 > m1.date1 OR (m2.date1 = m1.date1 AND m2.id > m1.id))"
            ")"
        )

        review_value_expr = (
            "CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))"
        )
        severity_sort_expr = (
            "CASE WHEN riv.review_direction = 1 THEN -" + review_value_expr + " ELSE " + review_value_expr + " END"
        )

        active_reviews_base = (
            "SELECT ri.instr_id, rl.review_name, " + review_value_expr + " AS review_value, riv.review_direction, " + severity_sort_expr + " AS severity_sort "
            "FROM review_instruments ri "
            "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "WHERE ri.review_field = %(review_field)s AND ri.review_status = 'ON'"
        )

        active_reviews_ranked = (
            "SELECT ar_base.instr_id, ar_base.review_name, ar_base.review_value, ar_base.review_direction, ar_base.severity_sort, "
            "(SELECT COUNT(*) + 1 FROM (" + active_reviews_base + ") AS ar2 WHERE ar2.instr_id = ar_base.instr_id AND ar2.severity_sort < ar_base.severity_sort) AS severity_rank "
            "FROM (" + active_reviews_base + ") AS ar_base"
        )

        inner_select = (
            "SELECT e.instr_id AS instrument_id, e.easting, e.northing, "
            "(SELECT CONCAT_WS('|', ar1.review_name, ar1.review_value) FROM (" + active_reviews_ranked + ") AS ar1 WHERE ar1.instr_id = e.instr_id "
            " AND ((ar1.review_direction = 1 AND sl.start_field_value > ar1.review_value) OR (ar1.review_direction = -1 AND sl.start_field_value < ar1.review_value)) "
            " ORDER BY ar1.severity_rank LIMIT 1) AS start_review_pair, "
            "sl.start_field_value AS start_db_field_value, sl.start_field_value_timestamp AS start_db_field_value_timestamp, "
            "(SELECT CONCAT_WS('|', ar1.review_name, ar1.review_value) FROM (" + active_reviews_ranked + ") AS ar1 WHERE ar1.instr_id = e.instr_id "
            " AND ((ar1.review_direction = 1 AND el.end_field_value > ar1.review_value) OR (ar1.review_direction = -1 AND el.end_field_value < ar1.review_value)) "
            " ORDER BY ar1.severity_rank LIMIT 1) AS end_review_pair, "
            "el.end_field_value AS end_db_field_value, el.end_field_value_timestamp AS end_db_field_value_timestamp "
            "FROM (" + eligible_sql + ") AS e "
            "JOIN (" + start_latest_sql + ") AS sl ON sl.instr_id = e.instr_id "
            "JOIN (" + end_latest_sql + ") AS el ON el.instr_id = e.instr_id"
        )

        sql = (
            "SELECT * FROM (" + inner_select + ") AS t "
            "WHERE NOT (SUBSTRING_INDEX(t.start_review_pair, '|', 1) <=> SUBSTRING_INDEX(t.end_review_pair, '|', 1)) "
            "ORDER BY t.end_db_field_value_timestamp DESC"
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

                def _split_pair(val: Any) -> Tuple[Optional[str], Optional[float]]:
                    if val is None:
                        return None, None
                    try:
                        status_part, value_part = str(val).split("|", 1)
                        return status_part or None, _coerce_float(value_part) if value_part != "" else None
                    except ValueError:
                        return str(val) or None, None

                start_status, s_rv = _split_pair(r.get("start_review_pair"))
                end_status, e_rv = _split_pair(r.get("end_review_pair"))

                if e is None or n is None or s_fv is None or e_fv is None:
                    raise ValueError("required numeric conversion failed")
                out.append(_MapSeriesReviewChangeRow(
                    instrument_id=r.get("instrument_id"),
                    easting=e,
                    northing=n,
                    start_review_status=start_status,
                    start_review_value=s_rv,
                    start_db_field_value=s_fv,
                    start_db_field_value_timestamp=r.get("start_db_field_value_timestamp"),
                    end_review_status=end_status,
                    end_review_value=e_rv,
                    end_db_field_value=e_fv,
                    end_db_field_value_timestamp=r.get("end_db_field_value_timestamp"),
                ).model_dump())
            except Exception as e:
                logger.warning("[get_map_series_review_change] Row skipped: %s row=%s", e, r)

        return pd.DataFrame(out) if out else None