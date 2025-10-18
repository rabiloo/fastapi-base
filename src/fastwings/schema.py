"""Schema definitions and utilities for FastAPI API requests and responses.

Classes:
    BaseRequestSchema: Base schema for API requests, supports configuration and field aliasing.
    Paging: Schema for paginated API requests.
    DateBetween: Schema for passing a date range between two dates.

Functions:
    all_optional: Creates a new Pydantic model with all fields being optional.
    ignore_numpy_fields: Creates a new Pydantic model with NumPy fields excluded.
"""

from datetime import datetime
from typing import Optional, TypeVar

from pydantic import BaseModel, ConfigDict, create_model, field_validator

from fastwings.model import BaseModel as DBBaseModel

SchemaInstance = TypeVar("SchemaInstance", bound=BaseModel)
ModelInstance = TypeVar("ModelInstance", bound=DBBaseModel)


class BaseRequestSchema(BaseModel):
    """Base schema for API requests, supports configuration and field aliasing.

    Provides:
        - Pydantic configuration for attribute population, assignment validation, enum value usage, and arbitrary types.
        - Utility method to collect field aliases for serialization/deserialization.
    """
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
    )

    @classmethod
    def collect_aliases(cls: type[BaseModel]) -> dict[str, str]:
        """Collects aliases for fields in the schema.

        Returns:
            Dictionary mapping alias names to real field names.
        """
        result = {}  # <alias_name>: <real_name> OR <real_name>: <real_name>
        for name, field in cls.model_fields.items():
            if field.alias:
                result.update({field.alias: name})
            else:
                result.update({name: name})
        return result


class Paging(BaseRequestSchema):
    """Paging schema for API requests.

    Attributes:
        offset (int | None): Start position.
        limit (int | None): Number of records to return.
    """
    offset: int | None = None
    limit: int | None = None


def all_optional(name: str, model: type[BaseModel]) -> type[BaseModel]:
    """Creates a new Pydantic model with all fields being optional.

    Args:
        name (str): The name for the new Pydantic model class.
        model (type[BaseModel]): The original Pydantic model class to modify.

    Returns:
        type[BaseModel]: New model class with all fields optional.
    """
    fields = {
        field_name: (Optional[field.annotation], None)  # noqa UP045
        for field_name, field in model.model_fields.items()
    }
    return create_model(name, **fields)  # type: ignore[call-overload, no-any-return]


def ignore_numpy_fields(name: str, model: type[BaseModel]) -> type[BaseModel]:
    """Creates a new Pydantic model that excludes fields with NumPy type annotations.

    This function iterates through the fields of the input model and builds a new model containing only those fields
    whose type annotation does not appear to be a NumPy type (by checking for 'numpy.' or 'npt.' in the type's string
    representation).

    Args:
        name (str): The name for the new Pydantic model class.
        model (type[BaseModel]): The original Pydantic model class to modify.

    Returns:
        type[BaseModel]: New model class with NumPy fields excluded.
    """
    new_fields = {}
    for field_name, field_info in model.model_fields.items():
        # Get the string representation of the field's type annotation.
        annotation_str = str(field_info.annotation)

        # If it's not a NumPy type, keep the field.
        if "numpy." not in annotation_str and "npt." not in annotation_str:
            # Recreate the field definition for the new model.
            new_fields[field_name] = (field_info.annotation, field_info.default)

    # Use create_model to dynamically generate a new model class.
    return create_model(name, **new_fields)  # type: ignore[call-overload, no-any-return]


class DateBetween(BaseModel):
    """Schema for passing a date range between two dates.

    Attributes:
        from_date: Start date.
        to_date: End date.
    """
    from_date: datetime
    to_date: datetime

    @field_validator("from_date", "to_date", mode='before')
    def parse_date(cls, value: str | datetime) -> datetime:
        """Parse date from string or datetime object.

        Args:
            value: Date value as string or datetime.

        Returns:
            datetime: Parsed datetime object.
        """
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return value
