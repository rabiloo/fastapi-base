from typing import Any, Generic, Self, TypeVar

from sqlalchemy import Column, Delete, Select, Update, delete, func, inspect, select, update
from sqlalchemy.sql.expression import ColumnElement

from fastwings.model import Base

ModelType = TypeVar("ModelType", bound=Base)


class QueryBuilder(Generic[ModelType]):
    """A chainable, mutable builder for creating SQLAlchemy 2.0+ queries.

    Each method call modifies the builder's internal state and returns the
    same instance (`self`) to allow for a fluent, chainable interface.

    Example:
        # Create a builder instance
        user_query_builder = QueryBuilder(User)

        # Chain methods to configure the query
        user_query_builder.add_filters(User.is_active == True) \
                          .order_by(User.created_at.desc()) \
                          .limit(10)

        # Generate the final SQLAlchemy statement
        statement = user_query_builder.as_select()

        # Execute the statement
        active_users = session.execute(statement).scalars().all()
    """

    def __init__(self, model_class: type[ModelType]):
        """Initializes the QueryBuilder for a specific model class.

        Args:
            model_class: The SQLAlchemy declarative model to build queries for.
        """
        self.model_class: type[ModelType] = model_class
        self._columns: list[ColumnElement] = []
        self._load_options: list[Any] = []
        self._filters: list[Any] = []
        self._filter_by: dict[str, Any] = {}
        self._joins: list[tuple[tuple, dict[str, Any]]] = []
        self._order_by: list[ColumnElement] = []
        self._group_by: list[ColumnElement] = []
        self._having: list[Any] = []
        self._limit: int | None = None
        self._offset: int | None = None
        self._distinct: bool = False
        self._distinct_on: list[ColumnElement] = []

    def select_columns(self, *columns: ColumnElement) -> Self:
        """Specify which columns to select instead of the entire model.

        Args:
            columns: Column expressions to select

        Example:
            QueryBuilder(User).select_columns(User.id, User.name)
        """
        self._columns = list(columns)
        return self

    def set_load_options(self, *options: Any) -> Self:
        """Sets relationship loading options, replacing any existing ones."""
        self._load_options = list(options)
        return self

    def add_load_options(self, *options: Any) -> Self:
        """Adds relationship loading options to the existing set."""
        self._load_options.extend(options)
        return self

    def set_filters(self, *filters: Any, **filter_by: Any) -> Self:
        """Sets filters, replacing any existing ones."""
        self._filters = list(filters)
        self._filter_by = filter_by
        return self

    def add_filters(self, *filters: Any, **filter_by: Any) -> Self:
        """Adds filters to the existing set."""
        self._filters.extend(filters)
        self._filter_by.update(filter_by)
        return self

    def join(self, *args: Any, **kwargs: Any) -> Self:
        """Adds a JOIN clause to the query."""
        self._joins.append((args, kwargs))
        return self

    def outerjoin(self, *args: Any, **kwargs: Any) -> Self:
        """Adds an OUTER JOIN (LEFT JOIN) clause to the query."""
        kwargs['isouter'] = True
        self._joins.append((args, kwargs))
        return self

    def where(self, *conditions: Any) -> Self:
        """Alias for add_filters() for more SQL-like syntax."""
        return self.add_filters(*conditions)

    def filter(self, *conditions: Any, **filter_by: Any) -> Self:
        """Alias for add_filters() for compatibility with SQLAlchemy Query API."""
        return self.add_filters(*conditions, **filter_by)

    def order_by(self, *clauses: ColumnElement) -> Self:
        """Sets the ORDER BY clause, accepting multiple sorting criteria."""
        self._order_by = list(clauses)
        return self

    def group_by(self, *clauses: ColumnElement) -> Self:
        """Sets the GROUP BY clause."""
        self._group_by = list(clauses)
        return self

    def having(self, *conditions: Any) -> Self:
        """Adds HAVING conditions (used with GROUP BY)."""
        self._having.extend(conditions)
        return self

    def limit(self, limit: int) -> Self:
        """Sets the LIMIT clause."""
        if limit < 0:
            raise ValueError("Limit must be a non-negative integer.")
        self._limit = limit
        return self

    def offset(self, offset: int) -> Self:
        """Sets the OFFSET clause."""
        if offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        self._offset = offset
        return self

    def paginate(self, page: int, per_page: int) -> Self:
        """Convenience method for pagination.

        Args:
            page: Page number (1-indexed)
            per_page: Number of items per page
        """
        if page < 1:
            raise ValueError("Page number must be 1 or greater.")
        if per_page < 1:
            raise ValueError("Items per page must be 1 or greater.")
        self._limit = per_page
        self._offset = (page - 1) * per_page
        return self

    def distinct(self, is_distinct: bool = True) -> Self:
        """Applies a DISTINCT clause to the select query."""
        self._distinct = is_distinct
        return self

    def distinct_on(self, *columns: ColumnElement) -> Self:
        """Applies DISTINCT ON clause (PostgreSQL specific).

        Args:
            columns: Columns to apply DISTINCT ON
        """
        self._distinct_on = list(columns)
        return self

    def _get_primary_key_column(self) -> Column:
        """Retrieves the primary key column of the model class.

        Returns:
            The primary key column

        Raises:
            ValueError: If no primary key or composite primary key exists
        """
        mapper = inspect(self.model_class)
        pk_columns = list(mapper.primary_key)

        if len(pk_columns) == 0:
            raise ValueError(f"{self.model_class.__name__} has no primary key defined")
        if len(pk_columns) > 1:
            raise ValueError(
                f"{self.model_class.__name__} has a composite primary key. "
                "Specify the column explicitly for count operations."
            )

        return pk_columns[0]

    def _apply_common_clauses(self, stmt: Select | Delete | Update) -> Select | Delete | Update:
        """Helper to apply filters and joins to a statement."""
        # Only apply joins to SELECT statements
        if isinstance(stmt, Select):
            for join_args, join_kwargs in self._joins:
                stmt = stmt.join(*join_args, **join_kwargs)

        # Apply filters
        stmt = stmt.where(*self._filters).filter_by(**self._filter_by)
        return stmt

    def as_select(self) -> Select:
        """Builds and returns a SELECT statement from the current state."""
        # Select specific columns if provided, otherwise select the entire model
        if self._columns:
            stmt = select(*self._columns)
            # Need to explicitly specify from_clause when selecting columns
            stmt = stmt.select_from(self.model_class)
        else:
            stmt = select(self.model_class)

        if self._distinct_on:
            stmt = stmt.distinct(*self._distinct_on)
        elif self._distinct:
            stmt = stmt.distinct()

        stmt = self._apply_common_clauses(stmt)

        # Load options only apply when selecting full entities
        if not self._columns:
            stmt = stmt.options(*self._load_options)

        if self._group_by:
            stmt = stmt.group_by(*self._group_by)

        if self._having:
            stmt = stmt.having(*self._having)

        if self._order_by:
            stmt = stmt.order_by(*self._order_by)

        if self._limit is not None:
            stmt = stmt.limit(self._limit)

        if self._offset is not None:
            stmt = stmt.offset(self._offset)

        return stmt

    def as_delete(self) -> Delete:
        """Builds and returns a DELETE statement from the current state.

        Note: DELETE statements don't support joins in most databases.
        """
        if self._joins:
            raise ValueError(
                "DELETE statements don't support JOIN clauses in most databases. "
                "Use subqueries or filter by IDs instead."
            )

        stmt = delete(self.model_class)
        stmt = stmt.where(*self._filters).filter_by(**self._filter_by)
        return stmt

    def as_update(self, values: dict[str, Any]) -> Update:
        """Builds and returns an UPDATE statement from the current state.

        Args:
            values: A dictionary of column names to new values.

        Raises:
            ValueError: If values is empty or joins are present
        """
        if not values:
            raise ValueError("Update values cannot be empty")

        if self._joins:
            raise ValueError(
                "UPDATE statements don't support JOIN clauses in most databases. "
                "Use subqueries or filter by IDs instead."
            )

        stmt = update(self.model_class).values(values)
        stmt = stmt.where(*self._filters).filter_by(**self._filter_by)
        return stmt

    def as_count(self, column: ColumnElement | None = None) -> Select:
        """Builds an efficient query to count results based on the current filters.

        Ignores ordering, limit, offset, and load options.

        Args:
            column: Optional column to count. If None, uses primary key.

        Returns:
            A SELECT statement that returns a count
        """
        count_target = column if column is not None else self._get_primary_key_column()

        # For DISTINCT, count distinct values of the column
        if self._distinct:
            count_expr = func.count(func.distinct(count_target))
        else:
            count_expr = func.count(count_target)

        stmt = select(count_expr).select_from(self.model_class)
        stmt = self._apply_common_clauses(stmt)

        # Apply group_by and having if present (for grouped counts)
        if self._group_by:
            stmt = stmt.group_by(*self._group_by)

        if self._having:
            stmt = stmt.having(*self._having)

        return stmt

    def as_exists(self) -> Select:
        """Builds an efficient EXISTS query based on current filters.

        Returns:
            A SELECT statement that returns True/False
        """
        subquery = select(1).select_from(self.model_class)
        subquery = self._apply_common_clauses(subquery)

        # EXISTS only needs to find one row, so always limit to 1 for performance
        subquery = subquery.limit(1)

        return select(subquery.exists())

    def as_scalar_subquery(self) -> Select:
        """Converts the current SELECT query into a scalar subquery.

        Useful for correlated subqueries in SELECT clauses or WHERE conditions.

        Returns:
            A scalar subquery that can be used in other queries
        """
        return self.as_select().scalar_subquery()

    def as_subquery(self, name: str | None = None) -> Select:
        """Converts the current SELECT query into a subquery/derived table.

        Args:
            name: Optional alias name for the subquery

        Returns:
            A subquery that can be joined or selected from
        """
        subq = self.as_select().subquery()
        if name:
            subq = subq.alias(name)
        return subq

    def clone(self) -> Self:
        """Creates a deep copy of the current QueryBuilder instance.

        Returns:
            A new QueryBuilder with the same configuration
        """
        import copy

        new_builder = self.__class__(self.model_class)

        # Deep copy all mutable attributes to avoid shared references
        new_builder._columns = copy.copy(self._columns)
        new_builder._load_options = copy.copy(self._load_options)
        new_builder._filters = copy.copy(self._filters)
        new_builder._filter_by = copy.copy(self._filter_by)
        new_builder._joins = copy.copy(self._joins)
        new_builder._order_by = copy.copy(self._order_by)
        new_builder._group_by = copy.copy(self._group_by)
        new_builder._having = copy.copy(self._having)
        new_builder._distinct_on = copy.copy(self._distinct_on)

        # Copy immutable attributes directly
        new_builder._limit = self._limit
        new_builder._offset = self._offset
        new_builder._distinct = self._distinct

        return new_builder

    def reset(self) -> Self:
        """Resets all query parameters to their initial state."""
        self._columns.clear()
        self._load_options.clear()
        self._filters.clear()
        self._filter_by.clear()
        self._joins.clear()
        self._order_by.clear()
        self._group_by.clear()
        self._having.clear()
        self._limit = None
        self._offset = None
        self._distinct = False
        self._distinct_on.clear()
        return self


class SoftDeletableQueryBuilder(QueryBuilder[ModelType]):
    """An enhanced QueryBuilder that automatically filters out soft-deleted records (where 'is_deleted' == True) for all read operations.

    This behavior can be disabled on a per-query basis by calling the .include_deleted() method.
    """

    def __init__(self, model_class: type[ModelType]):
        """Initializes the builder and verifies the model supports soft-delete.

        Raises:
            TypeError: If the model does not have an 'is_deleted' attribute.
        """
        super().__init__(model_class)

        if not hasattr(self.model_class, "is_deleted"):
            raise TypeError(
                f"Model {self.model_class.__name__} does not have an 'is_deleted' "
                "attribute and cannot be used with SoftDeletableQueryBuilder."
            )

        self._filter_soft_deleted = True

    def _apply_common_clauses(self, stmt: Select | Delete | Update) -> Select | Delete | Update:
        """Applies all user-defined filters AND the automatic soft-delete filter.

        This hook is used by as_select(), as_count(), and as_exists(),
        ensuring consistent filtering for all read queries.
        """
        # 1. Apply all filters from the parent class (join, where, filter_by)
        stmt = super()._apply_common_clauses(stmt)
        if self._filter_soft_deleted and isinstance(stmt, Select):
            stmt = stmt.where(self.model_class.is_deleted.is_(False))

        return stmt
