"""Base models and interfaces for the application."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """Base model class that all other models should inherit from."""
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModel:
        """Create an instance from a dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary."""
        return self.model_dump()


class TimestampMixin:
    """Mixin for models that include created/updated timestamps."""
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class BaseJob(BaseModel, TimestampMixin):
    """Base job model with common job fields."""

    id: str | None = None
    title: str | None = None
    company: str | None = None
    location: str | None = None
    description: str | None = None
    url: str | None = None
    source: str | None = None  # e.g., 'indeed', 'linkedin', 'jobspy'

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseJob:
        return cls(**data)


class BaseResume(BaseModel, TimestampMixin):
    """Base resume model."""
    id: str | None = None
    file_path: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    content: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseResume:
        return cls(**data)
