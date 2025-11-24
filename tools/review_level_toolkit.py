import logging
import re
import json
import datetime as dt
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any
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
    field_value: float = Field(..., description="Most recent field value breaching the review level.")
    field_value_timestamp: datetime = Field(..., description="Timestamp of the field value.")
    review_value: float = Field(..., description="Threshold value of the named review level.")


class GetReviewStatusByValueTool(BaseTool, _BaseQueryTool):
    """
    Returns the review level status (most severe breached review level name) for a given
    database field value at a specific instrument and field. Returns None if no review
    is breached or no reviews exist.

    Input:
    - instrument_id: str - The instrument ID (from instrum.instr_id)
    - db_field_name: str - The system field name (e.g., 'data1', 'calculation1')
    - db_field_value: float - The actual field value to evaluate against review thresholds

    Returns:
    - str: Review level name if breached, None if not, or ERROR: message if invalid.
    """
    name: str = "get_review_status_by_value_tool"
    description: str = (
        """
        Returns the review level status (most severe breached review level name) for a given
        database field value at a specific instrument and field. Returns None if no review
        is breached or no reviews exist.

        Input:
        - instrument_id: str - The instrument ID (from instrum.instr_id)
        - db_field_name: str - The system field name (e.g., 'data1', 'calculation1')
        - db_field_value: float - The actual field value to evaluate against review thresholds

        Returns:
        - str: Review level name if breached, None if not, or ERROR: message if invalid.
        """
    )

    def _run(self, instrument_id: str, db_field_name: str, db_field_value: float) -> Union[str, None]:
        if not instrument_id or not db_field_name:
            return "ERROR: instrument_id and db_field_name are required."
        try:
            db_field_value = float(db_field_value)
        except (TypeError, ValueError):
            return "ERROR: db_field_value must be numeric."

        params = {
            "instrument_id": instrument_id,
            "review_field": db_field_name,
        }
        sql = (
            "SELECT rl.review_name, riv.review_value, riv.review_direction "
            "FROM review_instruments ri "
            "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "WHERE ri.instr_id = %(instrument_id)s "
            "AND ri.review_field = %(review_field)s "
            "AND ri.review_status = 'ON' "
            "AND REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END"
        )
        rendered = sql % {k: _quote(v) for k, v in params.items()}
        try:
            logger.info("[get_review_status_from_value] SQL=%s", rendered)
            result = self.sql_tool._run(rendered)
            logger.info("[get_review_status_from_value] Raw result=%s", result)
        except Exception as e:
            return f"ERROR: query failed: {e}"
        if _is_no_data(result):
            return None
        try:
            rows = _parse_rows(result)
        except Exception as e:
            return f"ERROR: parse failed: {e}"
        if not rows:
            return None
        for row in rows:
            name = row.get("review_name")
            value = _coerce_float(row.get("review_value"))
            direction = _coerce_int(row.get("review_direction"))
            if value is None or direction is None:
                continue
            if direction == 1 and db_field_value > value:
                return name
            if direction == -1 and db_field_value < value:
                return name
        return None

class GetReviewStatusByTimeTool(BaseTool, _BaseQueryTool):
    """
    Finds the most recent reading before `timestamp` for `instrument_id` and `db_field_name`,
    then returns its review status, value, and timestamp.

    Returns a pandas DataFrame with one row (ReviewStatusOutput fields as columns), or None, or an ERROR message.
    """
    name: str = "get_review_status_by_time_tool"
    description: str = (
        """
        Finds the most recent reading before `timestamp` for `instrument_id` and `db_field_name`,
        then returns its review status, value, and timestamp.

        Returns a pandas DataFrame with one row (ReviewStatusOutput fields as columns), or None, or an ERROR message.
        """
    )

    def _run(self, instrument_id: str, db_field_name: str, timestamp: Union[str, datetime]) -> Union[pd.DataFrame, None, str]:
        if not instrument_id or not db_field_name:
            return "ERROR: instrument_id and db_field_name are required." 
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                return "ERROR: Invalid timestamp format (use ISO)."

        if db_field_name.startswith("calculation"):
            calc_num = db_field_name.replace("calculation", "")
            json_path = f"$.calculation{calc_num}"
            params = {"json_path": json_path, "instrument_id": instrument_id, "review_field": db_field_name, "ts": timestamp}
            sql = (
                "WITH latest_reading AS ( "
                "SELECT m.date1 AS reading_time, "
                "CASE WHEN JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "THEN CAST(REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END AS field_value "
                "FROM mydata m WHERE m.instr_id = %(instrument_id)s AND m.date1 < %(ts)s "
                "AND JSON_VALID(m.custom_fields) AND JSON_EXTRACT(m.custom_fields, %(json_path)s) IS NOT NULL "
                "AND REPLACE(NULLIF(TRIM(JSON_UNQUOTE(JSON_EXTRACT(m.custom_fields, %(json_path)s))), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "ORDER BY m.date1 DESC LIMIT 1), "
                "breached AS ( SELECT rl.review_name, riv.review_value, riv.review_direction FROM review_instruments ri "
                "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "JOIN review_levels rl ON riv.review_level_id = rl.id "
                "WHERE ri.instr_id = %(instrument_id)s AND ri.review_field = %(review_field)s AND ri.review_status='ON' "
                "AND REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "AND ((riv.review_direction = 1 AND (SELECT field_value FROM latest_reading) > CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))) "
                "OR (riv.review_direction = -1 AND (SELECT field_value FROM latest_reading) < CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)))) "
                "ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END LIMIT 1) "
                "SELECT lr.reading_time, lr.field_value, (SELECT review_name FROM breached) AS review_name FROM latest_reading lr"
            )
        else:
            try:
                valid_col = _validate_col(db_field_name)
            except ValueError as e:
                return f"ERROR: {e}" 
            params = {"instrument_id": instrument_id, "review_field": valid_col, "ts": timestamp}
            sql = (
                "WITH latest_reading AS ( "
                f"SELECT m.date1 AS reading_time, "
                f"CASE WHEN REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                f"THEN CAST(REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') AS DECIMAL(20,6)) ELSE NULL END AS field_value "
                f"FROM mydata m "
                f"WHERE m.instr_id = %(instrument_id)s AND m.date1 < %(ts)s AND REPLACE(NULLIF(TRIM(m.{valid_col}), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "ORDER BY m.date1 DESC LIMIT 1), "
                "breached AS ( SELECT rl.review_name, riv.review_value, riv.review_direction FROM review_instruments ri "
                "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
                "JOIN review_levels rl ON riv.review_level_id = rl.id "
                "WHERE ri.instr_id = %(instrument_id)s AND ri.review_field = %(review_field)s AND ri.review_status='ON' "
                "AND REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
                "AND ((riv.review_direction = 1 AND (SELECT field_value FROM latest_reading) > CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6))) "
                "OR (riv.review_direction = -1 AND (SELECT field_value FROM latest_reading) < CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)))) "
                "ORDER BY CASE WHEN riv.review_direction = 1 THEN -CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) ELSE CAST(REPLACE(NULLIF(TRIM(riv.review_value), ''), ',', '') AS DECIMAL(20,6)) END LIMIT 1) "
                "SELECT lr.reading_time, lr.field_value, (SELECT review_name FROM breached) AS review_name FROM latest_reading lr"
            )

        rendered = sql % {k: _quote(v) for k, v in params.items()}
        try:
            logger.info("[get_review_status_from_time0] SQL=%s", rendered)
            result = self.sql_tool._run(rendered)
            logger.info("[get_review_status_from_time0] Raw result=%s", result)
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
        row = rows[0]
        field_value = _coerce_float(row.get("field_value"))
        if field_value is None:
            return None
        review_name = row.get("review_name")
        record = _ReviewStatusOutput(
            review_status=review_name,
            db_field_value=field_value,
            db_field_value_timestamp=row.get("reading_time"),
        ).model_dump()
        return pd.DataFrame([record])

class GetReviewSchemaTool(BaseTool, _BaseQueryTool):
    """
    Returns all active review levels for an instrument and field, including name, value,
    direction, and color.

    Returns a pandas DataFrame with one row per review level (columns match ReviewLevelSchema), or None, or an ERROR.
    """
    name: str = "get_review_schema_tool"
    description: str = (
        """
        Returns all active review levels for an instrument and field, including name, value,
        direction, and color.

        Returns a pandas DataFrame with one row per review level (columns match ReviewLevelSchema), or None, or an ERROR.
        """
    )

    def _run(self, instrument_id: str, db_field_name: str) -> Union[pd.DataFrame, None, str]:
        if not instrument_id or not db_field_name:
            return "ERROR: instrument_id and db_field_name are required."
        params = {"instrument_id": instrument_id, "review_field": db_field_name}
        sql = (
            "SELECT rl.review_name, riv.review_value, CASE WHEN riv.review_direction = 1 THEN 'upper' ELSE 'lower' END AS review_direction, "
            "CONCAT('#', aci.aaa_color) AS review_color FROM review_instruments ri "
            "JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "JOIN aaa_color_info aci ON rl.id = aci.review_id "
            "WHERE ri.instr_id = %(instrument_id)s AND ri.review_field = %(review_field)s AND ri.review_status = 'ON' "
            "ORDER BY aci.`order`"
        )
        rendered = sql % {k: _quote(v) for k, v in params.items()}
        try:
            logger.info("[get_review_schema] SQL=%s", rendered)
            result = self.sql_tool._run(rendered)
            logger.info("[get_review_schema] Raw result=%s", result)
        except Exception as e:
            return f"ERROR: schema query failed: {e}"
        if _is_no_data(result):
            return None
        try:
            rows = _parse_rows(result)
        except Exception as e:
            return f"ERROR: schema parse failed: {e}"
        if not rows:
            return None
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(_ReviewLevelSchema(
                review_name=r.get("review_name"),
                review_value=_coerce_float(r.get("review_value")) or 0.0,
                review_direction=r.get("review_direction"),
                review_color=r.get("review_color")
            ).model_dump())
        return pd.DataFrame(out) if out else None

class GetReviewValueTool(BaseTool, _BaseQueryTool):
    """
    Given instrument_id, db_field_name, and review_name, returns the threshold value and direction.

    Returns a pandas DataFrame with one row (ReviewValueOutput fields as columns), or None, or an ERROR.
    """
    name: str = "get_review_value_tool"
    description: str = (
        """
        Given instrument_id, db_field_name, and review_name, returns the threshold value and direction.

        Returns a pandas DataFrame with one row (ReviewValueOutput fields as columns), or None, or an ERROR.
        """
    ) 

    def _run(self, instrument_id: str, db_field_name: str, review_name: str) -> Union[pd.DataFrame, None, str]:
        if not all([instrument_id, db_field_name, review_name]):
            return "ERROR: instrument_id, db_field_name, review_name required."
        params = {"instrument_id": instrument_id, "review_field": db_field_name, "review_name": review_name}
        sql = (
            "SELECT riv.review_value, CASE WHEN riv.review_direction = 1 THEN 'upper' ELSE 'lower' END AS review_direction "
            "FROM review_instruments ri JOIN review_instruments_values riv ON ri.id = riv.review_instr_id "
            "JOIN review_levels rl ON riv.review_level_id = rl.id "
            "WHERE ri.instr_id = %(instrument_id)s AND ri.review_field = %(review_field)s AND rl.review_name = %(review_name)s "
            "AND ri.review_status = 'ON' AND (ri.effective_from IS NULL OR ri.effective_from <= NOW()) LIMIT 1"
        )
        rendered = sql % {k: _quote(v) for k, v in params.items()}
        try:
            logger.info("[get_review_value] SQL=%s", rendered)
            result = self.sql_tool._run(rendered)
            logger.info("[get_review_value] Raw result=%s", result)
        except Exception as e:
            return f"ERROR: value query failed: {e}"
        if _is_no_data(result):
            return None
        try:
            rows = _parse_rows(result)
        except Exception as e:
            return f"ERROR: value parse failed: {e}"
        if not rows:
            return None
        r = rows[0]
        val = _coerce_float(r.get("review_value"))
        if val is None:
            return None
        record = _ReviewValueOutput(review_value=val, review_direction=r.get("review_direction")).model_dump()
        return pd.DataFrame([record])


class GetBreachedInstrumentsTool(BaseTool, _BaseQueryTool):
    """
    Finds instruments of given type/subtype where the latest reading before `timestamp`
    breaches the named review level, but does NOT breach any more severe level.

    Returns a pandas DataFrame with one row per breached instrument (columns match BreachedInstrumentReading), or None, or an ERROR.
    """
    name: str = "get_breached_instruments_tool"
    description: str = (
        """
        Finds instruments of given type/subtype where the latest reading before `timestamp`
        breaches the named review level, but does NOT breach any more severe level.

    Returns a pandas DataFrame with one row per breached instrument (columns match BreachedInstrumentReading), or None, or an ERROR.
        """
    )

    def _run(
        self,
        review_name: str,
        instrument_type: str,
        instrument_subtype: Optional[str],
        db_field_name: str,
        timestamp: Union[str, datetime],
    ) -> Union[pd.DataFrame, None, str]:
        if not all([review_name, instrument_type, db_field_name]):
            return "ERROR: review_name, instrument_type, and db_field_name are required."

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                return "ERROR: Invalid timestamp format. Use ISO format."

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

        sql = (
            "WITH latest_reading AS ( "
            "SELECT m.instr_id, m.date1 AS reading_time, " + field_expr + " AS field_value "
            "FROM mydata m WHERE m.date1 < %(ts)s AND " + reading_is_not_null + " ), "
            "ranked AS ( SELECT instr_id, reading_time, field_value, ROW_NUMBER() OVER (PARTITION BY instr_id ORDER BY reading_time DESC) AS rn FROM latest_reading ) "
            "SELECT r.instr_id AS instrument_id, r.field_value, r.reading_time AS field_value_timestamp, "
            "CAST(REPLACE(NULLIF(TRIM(target_riv.review_value), ''), ',', '') AS DECIMAL(20,6)) AS review_value "
            "FROM ranked r "
            "JOIN instrum i ON r.instr_id = i.instr_id "
            "JOIN review_instruments ri ON r.instr_id = ri.instr_id "
            "JOIN review_instruments_values target_riv ON ri.id = target_riv.review_instr_id "
            "JOIN review_levels target_rl ON target_riv.review_level_id = target_rl.id "
            "WHERE r.rn = 1 AND i.type1 = %(instrument_type)s "
            "AND (%(instrument_subtype)s IS NULL OR i.subtype1 = %(instrument_subtype)s) "
            "AND ri.review_field = %(review_field)s AND target_rl.review_name = %(review_name)s "
            "AND ri.review_status='ON' "
            "AND REPLACE(NULLIF(TRIM(target_riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "AND ((target_riv.review_direction = 1 AND r.field_value > CAST(REPLACE(NULLIF(TRIM(target_riv.review_value), ''), ',', '') AS DECIMAL(20,6))) "
            "OR (target_riv.review_direction = -1 AND r.field_value < CAST(REPLACE(NULLIF(TRIM(target_riv.review_value), ''), ',', '') AS DECIMAL(20,6)))) "
            "AND NOT EXISTS ( SELECT 1 FROM review_instruments_values s_riv JOIN review_levels s_rl ON s_riv.review_level_id = s_rl.id "
            "WHERE s_riv.review_instr_id = ri.id "
            "AND REPLACE(NULLIF(TRIM(s_riv.review_value), ''), ',', '') REGEXP '^-?[0-9]+(\\.[0-9]+)?$' "
            "AND ((s_riv.review_direction = 1 "
            "AND CAST(REPLACE(NULLIF(TRIM(s_riv.review_value), ''), ',', '') AS DECIMAL(20,6)) > CAST(REPLACE(NULLIF(TRIM(target_riv.review_value), ''), ',', '') AS DECIMAL(20,6)) "
            "AND r.field_value > CAST(REPLACE(NULLIF(TRIM(s_riv.review_value), ''), ',', '') AS DECIMAL(20,6))) "
            "OR (s_riv.review_direction = -1 "
            "AND CAST(REPLACE(NULLIF(TRIM(s_riv.review_value), ''), ',', '') AS DECIMAL(20,6)) < CAST(REPLACE(NULLIF(TRIM(target_riv.review_value), ''), ',', '') AS DECIMAL(20,6)) "
            "AND r.field_value < CAST(REPLACE(NULLIF(TRIM(s_riv.review_value), ''), ',', '') AS DECIMAL(20,6)))) ) "
            "ORDER BY r.reading_time DESC"
        )

        params: Dict[str, Any] = {
            "review_name": review_name,
            "instrument_type": instrument_type,
            "instrument_subtype": instrument_subtype,
            "review_field": valid_col,
            "ts": timestamp,
        }
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
                    field_value=fv,
                    field_value_timestamp=r.get("field_value_timestamp"),
                    review_value=rv,
                ).model_dump())
            except Exception as e:
                logger.warning("[get_breached_instruments] Row parse skipped: %s row=%s", e, r)
        logger.info("[get_breached_instruments] Parsed %d breached instruments", len(out))
        return pd.DataFrame(out) if out else None


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