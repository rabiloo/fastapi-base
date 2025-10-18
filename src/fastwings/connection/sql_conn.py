"""Implements synchronous database connection management for FastAPI applications.

Provides session manager and utilities for database health checks.
"""

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sqlalchemy.orm import Session, sessionmaker
from tenacity import retry, stop_after_attempt, wait_fixed

from fastwings.config import settings
from fastwings.error_code import ServerErrorCode

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages synchronous SQLAlchemy engine and session creation for FastAPI applications.

    Provides context managers for database connections and sessions.

    Args:
        host (str): Database connection string.
        engine_kwargs (dict, optional): Additional engine configuration.
    """

    def __init__(self, host: str, engine_kwargs: dict[str, Any] | None = None) -> None:
        """Initializes the SessionManager with engine and sessionmaker.

        Args:
            host (str): Database connection string.
            engine_kwargs (dict, optional): Additional engine configuration.
        """
        if engine_kwargs is None:
            engine_kwargs = {}
        self._engine: Engine | None = create_engine(host, **engine_kwargs)
        self._sessionmaker: sessionmaker[Session] | None = sessionmaker(autocommit=False, autoflush=False,
                                                                        bind=self._engine)

    def close(self) -> None:
        """Closes the engine and sessionmaker, releasing resources.

        Raises:
            Exception: If SessionManager is not initialized.
        """
        if self._engine is None:
            raise Exception("SessionManager is not initialized")
        self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    @contextlib.contextmanager
    def connect(self) -> Iterator[Connection]:
        """Context manager for database connection.

        Yields:
            Connection: An active database connection.

        Raises:
            Exception: If SessionManager is not initialized.
        """
        if self._engine is None:
            raise Exception("SessionManager is not initialized")

        with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        """Context manager for database session.

        Yields:
            Session: An active database session.

        Raises:
            Exception: If SessionManager is not initialized.
        """
        if self._sessionmaker is None:
            raise Exception("SessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception as ex:
            session.rollback()
            raise ServerErrorCode.DATABASE_ERROR.value(ex) from ex
        finally:
            session.close()


if settings.DB_ENGINE == "postgre":
    engine = "postgresql+psycopg2"
elif settings.DB_ENGINE == "mysql":
    engine = "mysql+pymysql"
else:
    raise ValueError(f"Not support for engine: {settings.DB_ENGINE}")

str_connection = f"{engine}://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/{settings.DB_NAME}"
sessionmanager = SessionManager(
    str_connection, engine_kwargs={"pool_pre_ping": True, "pool_size": settings.DB_POOL_SIZE}
)


def get_db_session() -> Iterator[Session]:
    """Provides a generator for yielding a database session using SessionManager.

    Yields:
        Session: An active database session.
    """
    with sessionmanager.session() as session:
        yield session


@retry(
    stop=stop_after_attempt(30),
    wait=wait_fixed(1),
)
def is_database_online() -> bool:
    """Checks if the database is online by executing a simple query.

    Returns:
        bool: True if database is online, False otherwise.
    """
    try:
        for session in get_db_session():
            with session:
                session.execute(select(1))
    except (SQLAlchemyError, TimeoutError):
        return False
    return True


if __name__ == "__main__":
    is_database_online()
