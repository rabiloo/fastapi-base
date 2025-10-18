"""Implements synchronous CRUD repository for FastAPI applications.

Provides methods for create, read, update, and delete operations using SQLAlchemy.

Classes:
    SQLRepository: Synchronous CRUD repository for SQLAlchemy models.
    SoftDeletableRepository: CRUD repository supporting soft deletion for SQLAlchemy models.
"""

import logging
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy import ColumnElement, select
from sqlalchemy.orm import Session

from fastwings.crud.sql_query_builder import QueryBuilder, SoftDeletableQueryBuilder
from fastwings.model import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=PydanticBaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=PydanticBaseModel)

logger = logging.getLogger(__name__)


class SQLRepository(Generic[ModelType]):
    """Synchronous CRUD repository for SQLAlchemy models.

    Provides methods for create, read, update, and delete operations.

    Attributes:
        model (type[ModelType]): SQLAlchemy model class managed by the repository.
        model_id_column (ColumnElement[Any]): Column used for default ordering and identification.
    """

    def __init__(self, model: type[ModelType]):
        """Initializes the repository with a SQLAlchemy model class.

        Args:
            model (type[ModelType]): SQLAlchemy model class.
        """
        self.model = model
        self.model_id_column: ColumnElement[Any] = self.model.id

    def query(self) -> QueryBuilder[ModelType]:
        """Creates a new QueryBuilder instance for this repository's model.

        Returns:
            QueryBuilder[ModelType]: A new query builder instance.

        Example:
            users = repo.query() \
                .add_filters(User.is_active == True) \
                .order_by(User.created_at.desc()) \
                .limit(10) \
                .all(session)
        """
        return QueryBuilder(self.model)

    def get(self, session: Session, obj_id: Any) -> ModelType | None:
        """Retrieves an object by its ID.

        Args:
            session (Session): SQLAlchemy session.
            obj_id (Any): Object ID to query.

        Returns:
            ModelType | None: Retrieved model instance or None if not found.
        """
        stmt = select(self.model).where(self.model.id == obj_id)
        result = session.execute(stmt)
        data = result.scalars().first()

        logger.debug(f"Get id: {obj_id} from table {self.model.__tablename__.upper()} done")
        return data

    def get_by(
        self,
        session: Session,
        *,
        order_by: Sequence[ColumnElement[Any]] | None = None,
        **filters: Any
    ) -> ModelType | None:
        """Retrieves a single object by filter conditions.

        Args:
            session (Session): SQLAlchemy session.
            order_by (Sequence[ColumnElement[Any]] | None, optional): Columns to order by. Defaults to model ID column.
            **filters: Filter conditions (e.g., email="user@example.com").

        Returns:
            ModelType | None: Retrieved model instance or None if not found.
        """
        stmt = self.query() \
            .add_filters(**filters) \
            .order_by(*(order_by if order_by is not None else [self.model_id_column])) \
            .limit(1) \
            .as_select()
        result = session.execute(stmt)
        data = result.scalars().first()

        logger.debug(f"Get by {filters} from table {self.model.__tablename__.upper()} done")
        return data

    def get_multi(
        self,
        session: Session,
        *,
        offset: int = 0,
        limit: int = 100,
        order_by: Sequence[ColumnElement[Any]] | None = None,
        **filters: Any
    ) -> Sequence[ModelType]:
        """Retrieves multiple objects with optional pagination and filters.

        Args:
            session (Session): SQLAlchemy session.
            offset (int, optional): Number of records to skip. Defaults to 0.
            limit (int, optional): Maximum number of records to return. Defaults to 100.
            order_by (Sequence[ColumnElement[Any]] | None, optional): Columns to order by. Defaults to model ID column.
            **filters: Filter conditions.

        Returns:
            Sequence[ModelType]: List of model instances.
        """
        stmt = self.query() \
            .add_filters(**filters) \
            .order_by(*(order_by if order_by is not None else [self.model_id_column])) \
            .offset(offset) \
            .limit(limit) \
            .as_select()
        result = session.execute(stmt)
        data = result.scalars().all()

        logger.debug(
            f"Get multi (offset={offset}, limit={limit}) from table {self.model.__tablename__.upper()} done"
        )
        return data

    def get_all(
        self,
        session: Session,
        *,
        order_by: Sequence[ColumnElement[Any]] | None = None,
        **filters: Any
    ) -> Sequence[ModelType]:
        """Retrieves all objects matching the given filters.

        Args:
            session (Session): SQLAlchemy session.
            order_by (Sequence[ColumnElement[Any]] | None, optional): Columns to order by. Defaults to model ID column.
            **filters: Filter conditions.

        Returns:
            Sequence[ModelType]: List of all matching model instances.
        """
        stmt = self.query() \
            .add_filters(**filters) \
            .order_by(*(order_by if order_by is not None else [self.model_id_column])) \
            .as_select()
        result = session.execute(stmt)
        data = result.scalars().all()

        logger.debug(f"Get all from table {self.model.__tablename__.upper()} done")
        return data

    def count(
        self,
        session: Session,
        **filters: Any
    ) -> int:
        """Counts objects matching the given filters.

        Args:
            session (Session): SQLAlchemy session.
            **filters: Filter conditions.

        Returns:
            int: Number of matching records.
        """
        stmt = self.query().add_filters(**filters).as_count()
        result = session.execute(stmt)
        count = cast(int, result.scalar_one())

        logger.debug(f"Count from table {self.model.__tablename__.upper()}: {count}")
        return count

    def exists(
        self,
        session: Session,
        **filters: Any
    ) -> bool:
        """Checks if any object exists matching the given filters.

        Args:
            session (Session): SQLAlchemy session.
            **filters: Filter conditions.

        Returns:
            bool: True if at least one matching record exists, False otherwise.
        """
        stmt = self.query().add_filters(**filters).as_exists()
        result = session.execute(stmt)
        exists = cast(bool, result.scalar_one())

        logger.debug(f"Exists check in table {self.model.__tablename__.upper()}: {exists}")
        return exists

    def create(
        self,
        session: Session,
        *,
        obj_in: CreateSchemaType,
    ) -> ModelType:
        """Creates a new object in the database.

        Args:
            session (Session): SQLAlchemy session.
            obj_in (CreateSchemaType): Pydantic schema for creation.

        Returns:
            ModelType: Created model instance.
        """
        obj_in_data = obj_in.model_dump(exclude_unset=True)
        db_obj = self.model(**obj_in_data)  # type: ignore
        session.add(db_obj)

        session.flush([db_obj])
        session.refresh(db_obj)

        logger.debug(f"Insert to table {self.model.__tablename__.upper()} done")
        return db_obj

    def create_multi(
        self,
        session: Session,
        *,
        objs_in: list[CreateSchemaType],
    ) -> list[ModelType]:
        """Creates multiple objects in a single transaction.

        Args:
            session (Session): SQLAlchemy session.
            objs_in (list[CreateSchemaType]): List of creation schemas.

        Returns:
            list[ModelType]: List of created model instances.
        """
        db_objs = [
            self.model(**obj_in.model_dump(exclude_unset=True))  # type: ignore
            for obj_in in objs_in
        ]
        session.add_all(db_objs)
        session.flush(db_objs)

        logger.debug(f"Bulk insert {len(db_objs)} records to table {self.model.__tablename__.upper()} done")
        return db_objs

    def update(
        self,
        session: Session,
        *,
        obj_id: Any,
        obj_in: UpdateSchemaType | dict[str, Any],
    ) -> ModelType:
        """Updates an object in the database.

        Args:
            session (Session): SQLAlchemy session.
            obj_id (Any): Object ID to update.
            obj_in (UpdateSchemaType | dict[str, Any]): Update data.

        Returns:
            ModelType: Updated model instance.

        Raises:
            ValueError: If the object with the given ID is not found.
        """
        obj = self.get(session, obj_id)
        if obj is None:
            raise ValueError(f"Object with id {obj_id} not found.")

        update_data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(obj, key, value)

        session.flush([obj])

        logger.debug(f"Update in table {self.model.__tablename__.upper()} done")
        return obj

    def update_multi(
        self,
        session: Session,
        *,
        values: dict[str, Any],
        **filters: Any
    ) -> int:
        """Updates multiple records matching the filters.

        Args:
            session (Session): SQLAlchemy session.
            values (dict[str, Any]): Values to update.
            **filters: Filter conditions to select records to update.

        Returns:
            int: Number of updated records.
        """
        stmt = self.query().add_filters(**filters).as_update(values)
        result = session.execute(stmt)

        session.flush()
        updated_count = result.rowcount

        logger.debug(f"Bulk update {updated_count} records in table {self.model.__tablename__.upper()} done")
        return updated_count

    def delete(
        self,
        session: Session,
        *,
        obj_id: Any,
    ) -> None:
        """Deletes an object in the database.

        Args:
            session (Session): SQLAlchemy session.
            obj_id (Any): Object ID to delete.

        Raises:
            ValueError: If the object with the given ID is not found.
        """
        obj = self.get(session, obj_id)
        if obj is None:
            raise ValueError(f"Object with id {obj_id} not found.")

        session.delete(obj)
        session.flush()

        logger.debug(f"Delete {obj_id} from table {self.model.__tablename__.upper()} done")

    def delete_multi(
        self,
        session: Session,
        **filters: Any
    ) -> int:
        """Deletes multiple records matching the filters.

        Args:
            session (Session): SQLAlchemy session.
            **filters: Filter conditions to select records to delete.

        Returns:
            int: Number of deleted records.
        """
        stmt = self.query().add_filters(**filters).as_delete()
        result = session.execute(stmt)

        session.flush()
        deleted_count = result.rowcount

        logger.debug(f"Bulk delete {deleted_count} records from table {self.model.__tablename__.upper()} done")
        return deleted_count

    def paginate(
        self,
        session: Session,
        *,
        page: int = 1,
        per_page: int = 20,
        order_by: Sequence[ColumnElement[Any]] | None = None,
        **filters: Any
    ) -> tuple[Sequence[ModelType], int]:
        """Paginates results with total count.

        Args:
            session (Session): SQLAlchemy session.
            page (int, optional): Page number (1-indexed). Defaults to 1.
            per_page (int, optional): Number of items per page. Defaults to 20.
            order_by (Sequence[ColumnElement[Any]] | None, optional): Columns to order by. Defaults to model ID column.
            **filters: Filter conditions.

        Returns:
            tuple[Sequence[ModelType], int]: Tuple of (items, total_count).
        """
        # Get paginated items
        items_stmt = self.query() \
            .add_filters(**filters) \
            .order_by(*(order_by if order_by is not None else [self.model_id_column])) \
            .paginate(page, per_page) \
            .as_select()

        items_result = session.execute(items_stmt)
        items = items_result.scalars().all()

        # Get total count
        count_stmt = self.query().add_filters(**filters).as_count()
        count_result = session.execute(count_stmt)
        total = count_result.scalar_one()

        logger.debug(
            f"Paginate (page={page}, per_page={per_page}) from table {self.model.__tablename__.upper()} done"
        )
        return items, total

    def upsert(
        self,
        session: Session,
        *,
        obj_in: CreateSchemaType | UpdateSchemaType,
        match_fields: list[str],
    ) -> tuple[ModelType, bool]:
        """Inserts or updates based on matching fields.

        Args:
            session (Session): SQLAlchemy session.
            obj_in (CreateSchemaType | UpdateSchemaType): Data to insert or update.
            match_fields (list[str]): Fields to match for existing record.

        Returns:
            tuple[ModelType, bool]: Tuple of (model_instance, was_created).
        """
        obj_data = obj_in.model_dump(exclude_unset=True)

        # Build filters from match fields
        filters = {field: obj_data[field] for field in match_fields if field in obj_data}

        # Try to find existing record
        existing = self.get_by(session, **filters)

        if existing:
            # Update existing
            for key, value in obj_data.items():
                setattr(existing, key, value)
            db_obj = existing
            was_created = False
        else:
            # Create new
            db_obj = self.model(**obj_data)  # type: ignore
            session.add(db_obj)
            was_created = True

        session.flush([db_obj])
        session.refresh(db_obj)

        action = "Created" if was_created else "Updated"
        logger.debug(f"{action} in table {self.model.__tablename__.upper()} done")
        return db_obj, was_created


class SoftDeletableRepository(SQLRepository[ModelType]):
    """CRUD repository supporting soft deletion for SQLAlchemy models.

    Extends SQLRepository to provide utilities for soft-deletable models, allowing records to be marked as deleted
    instead of being removed from the database.
    """

    def query(self) -> SoftDeletableQueryBuilder[ModelType]:
        """Creates a new SoftDeletableQueryBuilder instance for this repository's model.

        Returns:
            SoftDeletableQueryBuilder[ModelType]: A new query builder instance supporting soft deletion.
        """
        return SoftDeletableQueryBuilder(self.model)

    def delete(
        self,
        session: Session,
        *,
        obj_id: Any,
    ) -> None:
        """Marks an object as deleted in the database (soft delete).

        Args:
            session (Session): SQLAlchemy session.
            obj_id (Any): Object ID to delete.
        """
        obj = self.get(session, obj_id)
        # Add None check for type safety and to prevent AttributeError
        if obj is None:
            raise ValueError(f"Object with id {obj_id} not found.")

        obj.soft_delete()
        session.flush([obj])

        logger.debug(f"Delete {obj_id} from table {self.model.__tablename__.upper()} done")

    def delete_multi(
        self,
        session: Session,
        **filters: Any
    ) -> int:
        """Soft-deletes multiple records matching the filters.

        Args:
            session (Session): SQLAlchemy session.
            **filters: Filter conditions to select records to delete.

        Returns:
            int: Number of deleted records.
        """
        # Note: self.get_all() uses self.query(), which is SoftDeletableQueryBuilder,
        # so this will only fetch non-deleted items by default.
        objs = self.get_all(session, **filters)
        if not objs:
            return 0

        for obj in objs:
            obj.soft_delete()
        session.flush(objs)

        logger.debug(f"Bulk delete {len(objs)} records from table {self.model.__tablename__.upper()} done")

        return len(objs)
