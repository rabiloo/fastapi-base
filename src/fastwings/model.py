"""Defines the base SQLAlchemy model and utility functions for ORM usage.

Provides automatic table naming and dictionary conversion for model instances.

Classes:
    Base: Base SQLAlchemy model with automatic tablename and dict conversion.
"""

import re
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, ClassVar

import inflect
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy import String, Text, event, func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, Mapper
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import expression
from sqlmodel import Field, SQLModel

from fastwings.error_code import ServerErrorCode


# --- Core Model Interface and Lifecycle Events ---
class IDbModel:
    """An interface defining lifecycle hooks and field-level permissions for database models.

    These methods are automatically triggered by SQLAlchemy event listeners.
    """
    # Using ClassVar to indicate these are class-level configurations
    view_only_fields: ClassVar[Iterable[str]] = ()
    not_update_fields: ClassVar[Iterable[str]] = ()

    def before_create(self):
        """Called once before a new model instance is inserted into the database."""
        pass

    def before_update(self):
        """Called once before an existing model instance is updated in the database."""
        pass

    def before_save(self):
        """Called before both create and update operations."""
        pass

    def before_delete(self):
        """Called before a model instance is deleted from the database."""
        pass


# SQLAlchemy event listeners (remain unchanged, they are a good pattern)
@event.listens_for(IDbModel, 'before_insert', propagate=True)
def receive_before_insert(mapper, connection, target: IDbModel):
    target.before_create()
    target.before_save()


@event.listens_for(IDbModel, 'before_update', propagate=True)
def receive_before_update(mapper, connection, target: IDbModel):
    target.before_update()
    target.before_save()


@event.listens_for(IDbModel, 'before_delete', propagate=True)
def receive_before_delete(mapper, connection, target: IDbModel):
    target.before_delete()


def generate_unique_uuid() -> uuid.UUID:
    return uuid.uuid4()


def _validate_column_data(column, value: Any):
    """Validates data for a specific column based on its type and constraints."""
    # This logic is kept from your original code.
    if value is None:
        if not column.nullable and not column.primary_key and not column.server_default and not column.default:
            raise ServerErrorCode.DATABASE_ERROR.value(ValueError(f"'{column.name}' is a required field."))
        return

    if isinstance(column.type, (String, Text)) and not isinstance(value, str):
        # Optional: Add type checking if desired.
        pass

    if (isinstance(column.type, String)
        and hasattr(column.type, 'length')
        and column.type.length
        and len(str(value)) > column.type.length):
        raise ServerErrorCode.DATABASE_ERROR.value(
            ValueError(f"Maximum length reached for '{column.name}'. "
                       f"Must be {column.type.length} characters or less.")
        )


p = inflect.engine()


class DbModel(SQLModel, IDbModel):
    """"""

    id: Mapped[uuid.UUID] = Field(default_factory=generate_unique_uuid, primary_key=True, index=True, nullable=False)

    # Define base view-only and non-updatable fields
    view_only_fields: ClassVar[Iterable[str]] = ('id',)
    not_update_fields: ClassVar[Iterable[str]] = ('id',)

    @declared_attr  # type: ignore
    def __tablename__(cls) -> str:
        """Automatically generates table name from class name in plural snake_case.

        Returns:
            str: Table name for the model.
        """
        words = re.findall("[A-Z][^A-Z]*", cls.__name__)
        if len(words) == 1:
            return p.plural(words[0].lower())
        elif len(words) > 1:
            return "_".join(words[:-1]).lower() + "_" + p.plural(words[-1].lower())
        return cls.__name__.lower()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"

    def update(self, data: dict[str, Any] | PydanticBaseModel):
        """Updates the model instance with data from a dictionary or Pydantic model.

        It respects `view_only_fields` and `not_update_fields`.
        """
        if isinstance(data, PydanticBaseModel):
            # Use Pydantic's optimized method to get only set fields
            data = data.model_dump(exclude_unset=True)

        # Combine fields that should never be written to during an update
        protected_fields = set(self.view_only_fields) | set(self.not_update_fields)

        for key, value in data.items():
            # Check if the field is part of the model and not protected
            if hasattr(self, key) and key not in protected_fields:
                column = self.__table__.columns.get(key)
                if column is not None:
                    _validate_column_data(column, value)
                setattr(self, key, value)
        return self

    @classmethod
    def from_data(cls, data: dict[str, Any] | PydanticBaseModel):
        """Creates a new model instance from a dictionary or Pydantic model.

        It respects `view_only_fields`.
        """
        if isinstance(data, PydanticBaseModel):
            data = data.model_dump()

        # Filter out any data for fields that are view-only
        allowed_data = {
            key: value for key, value in data.items() if key not in cls.view_only_fields
        }

        # Validate data before creating the instance
        for key, value in allowed_data.items():
            column = cls.__table__.columns.get(key)
            if column is not None:
                _validate_column_data(column, value)

        return cls(**allowed_data)


@event.listens_for(Mapper, 'mapper_configured')
def validate_class_columns(mapper, cls):
    """Validates that any `view_only` field that isn't nullable has a default value.

    This prevents runtime errors on creation.
    """
    if not issubclass(cls, IDbModel):
        return

    for c in cls.__table__.columns:
        if c.primary_key:
            continue
        if c.name in cls.view_only_fields and not c.nullable and not c.server_default and not c.default:
            raise TypeError(
                f"Field '{c.name}' in model '{cls.__name__}' is 'view_only' and not nullable, "
                "but has no default or server_default value. This will cause errors on insert."
            )


class AuditableDbModel(IDbModel):
    """Mixin DbModel with tracking columns."""
    created_at: Mapped[datetime] = Field(
        description="Created time",
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": func.now()}
    )
    created_by: Mapped[int | None] = Field(default=None, description="The creator id", nullable=True)
    updated_at: Mapped[datetime | None] = Field(default=None, description="Last updated time", nullable=True)
    updated_by: Mapped[int | None] = Field(default=None, description="The latest updator id", nullable=True)

    def before_create(self):
        super().before_create()

        # Lấy session từ chính đối tượng này
        session = Session.object_session(self)
        if session and "user_id" in session.info:
            self.created_by = session.info["user_id"]

    def before_update(self):
        super().before_update()

        self.updated_at = datetime.now(timezone.utc)

        # Lấy session từ chính đối tượng này
        session = Session.object_session(self)
        if session and "user_id" in session.info:
            self.updated_by = session.info["user_id"]

    view_only_fields: ClassVar[Iterable[str]] = ('created_at', 'created_by', 'updated_at', 'updated_by')
    ignore_fields: ClassVar[Iterable[str]] = ('created_at', 'created_by', 'updated_at', 'updated_by')


class SoftDeletableDbModel(IDbModel):
    """Mixin DbModel with soft delete support."""

    is_deleted: bool = Field(
        default=False, sa_column_kwargs={"server_default": expression.false()}
    )

    view_only_fields: ClassVar[Iterable[str]] = ('is_deleted',)
    ignore_fields: ClassVar[Iterable[str]] = ('is_deleted',)

    def soft_delete(self):
        self.is_deleted = True


class BaseModel(AuditableDbModel, SoftDeletableDbModel, DbModel):
    """Base db model with audit fields and is_deleted field."""
