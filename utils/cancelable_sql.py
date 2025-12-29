from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

from langchain_community.utilities import SQLDatabase
from langchain_community.utilities.sql_database import sanitize_schema
from sqlalchemy.sql.elements import Executable
from sqlalchemy.sql.expression import text

from utils.run_cancellation import RunCancellationController, get_active_run_controller


class CancelableSQLDatabase(SQLDatabase):
    """SQLDatabase wrapper that cooperates with the cancellation controller."""

    def __init__(
        self,
        *args: Any,
    controller_getter: Optional[Callable[[], Optional[RunCancellationController]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._controller_getter = controller_getter or get_active_run_controller

    def _execute(
        self,
        command: Union[str, Executable],
        fetch: Literal["all", "one", "cursor"] = "all",
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[Sequence[Dict[str, Any]], Any]:
        controller = self._controller_getter() if self._controller_getter else None
        if controller:
            controller.raise_if_cancelled("sql:prepare")
        parameters = parameters or {}
        execution_options = execution_options or {}
        handle: Optional[str] = None
        connection = None
        try:
            with self._engine.begin() as connection:  # type: ignore[attr-defined]
                if controller:
                    handle = controller.register_sql_connection(connection, label=f"sql:{self.dialect}")
                if self._schema is not None:
                    if self.dialect == "snowflake":
                        connection.exec_driver_sql(
                            "ALTER SESSION SET search_path = %s",
                            (self._schema,),
                            execution_options=execution_options,
                        )
                    elif self.dialect == "bigquery":
                        connection.exec_driver_sql(
                            "SET @@dataset_id=?",
                            (self._schema,),
                            execution_options=execution_options,
                        )
                    elif self.dialect == "mssql":
                        pass
                    elif self.dialect == "trino":
                        connection.exec_driver_sql(
                            "USE ?",
                            (self._schema,),
                            execution_options=execution_options,
                        )
                    elif self.dialect == "duckdb":
                        connection.exec_driver_sql(
                            f"SET search_path TO {self._schema}",
                            execution_options=execution_options,
                        )
                    elif self.dialect == "oracle":
                        connection.exec_driver_sql(
                            f"ALTER SESSION SET CURRENT_SCHEMA = {self._schema}",
                            execution_options=execution_options,
                        )
                    elif self.dialect == "sqlany":
                        pass
                    elif self.dialect == "postgresql":
                        connection.exec_driver_sql(
                            "SET search_path TO %s",
                            (self._schema,),
                            execution_options=execution_options,
                        )
                    elif self.dialect == "hana":
                        connection.exec_driver_sql(
                            f"SET SCHEMA {sanitize_schema(self._schema)}",
                            execution_options=execution_options,
                        )

                if isinstance(command, str):
                    command = text(command)
                elif isinstance(command, Executable):
                    pass
                else:
                    raise TypeError(f"Query expression has unknown type: {type(command)}")

                cursor = connection.execute(
                    command,
                    parameters,
                    execution_options=execution_options,
                )

                if cursor.returns_rows:
                    if controller:
                        controller.raise_if_cancelled("sql:fetch")
                    if fetch == "all":
                        result = [x._asdict() for x in cursor.fetchall()]
                    elif fetch == "one":
                        first_result = cursor.fetchone()
                        result = [] if first_result is None else [first_result._asdict()]
                    elif fetch == "cursor":
                        return cursor
                    else:
                        raise ValueError("Fetch parameter must be either 'one', 'all', or 'cursor'")
                    return result
                return []
        finally:
            if controller and handle:
                controller.unregister(handle)
            if connection and controller and controller.is_cancelled():
                try:
                    connection.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
