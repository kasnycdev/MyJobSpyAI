"""Enhanced job analysis models for MyJobSpy AI."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class MatchStrength(str, Enum):
    """Enumeration of match strength levels."""
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXCELLENT = "excellent"


class SectionScore(BaseModel):
    """Model representing a score for a specific resume/job section."""

    score: float = Field(0.0, ge=0.0, le=100.0, description="Score from 0 to 100")
    weight: float = Field(
        1.0, ge=0.0, le=1.0, description="Weight of this section in the total score"
    )
    strength: MatchStrength = Field(
        MatchStrength.NONE, description="Qualitative assessment of the match"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )
    matched: list[str] = Field(
        default_factory=list, description="List of matched items in this section"
    )
    missing: list[str] = Field(
        default_factory=list, description="List of missing items in this section"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggested improvements for this section"
    )


class SectionScores(BaseModel):
    """Container for all section scores."""

    skills: SectionScore = Field(
        default_factory=lambda: SectionScore(weight=0.4),
        description="Skills match analysis",
    )
    experience: SectionScore = Field(
        default_factory=lambda: SectionScore(weight=0.3),
        description="Experience match analysis",
    )
    education: SectionScore = Field(
        default_factory=lambda: SectionScore(weight=0.15),
        description="Education match analysis",
    )
    certifications: SectionScore = Field(
        default_factory=lambda: SectionScore(weight=0.05),
        description="Certifications match analysis",
    )
    other: SectionScore = Field(
        default_factory=lambda: SectionScore(weight=0.1),
        description="Other factors match analysis",
    )

    @property
    def overall_score(self) -> float:
        """Calculate the weighted overall score."""
        fields = [
            f for f in dir(self)
            if not f.startswith('_') and not callable(getattr(self, f))
        ]
        total_weight = sum(
            getattr(self, field).weight
            for field in fields
            if hasattr(getattr(self, field), 'weight')
        )

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            getattr(self, field).score * getattr(self, field).weight
            for field in fields
            if hasattr(getattr(self, field, None), 'score') and \
               hasattr(getattr(self, field, None), 'weight')
        )

        return min(100.0, weighted_sum / total_weight if total_weight > 0 else 0.0)


class ImprovementRecommendation(BaseModel):
    """Model representing a recommendation for improvement."""

    category: str = Field(..., description="Category of the recommendation")
    title: str = Field(..., description="Title of the recommendation")
    description: str = Field(..., description="Detailed description")
    priority: int = Field(
        2, ge=1, le=3, description="Priority level (1=high, 2=medium, 3=low)"
    )
    action_items: list[str] = Field(
        default_factory=list, description="Specific actions to take"
    )
    resources: list[dict[str, str]] = Field(
        default_factory=list, description="List of resources with name and URL"
    )


class CoverLetterTemplate(BaseModel):
    """Model representing a cover letter template."""

    id: str = Field(..., description="Unique identifier for the template")
    name: str = Field(..., description="Template name")
    content: str = Field(..., description="Template content with placeholders")
    style: str = Field("professional", description="Template style")
    is_default: bool = Field(False, description="Whether this is the default template")


class TrainingResource(BaseModel):
    """Model representing a training or learning resource."""

    id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Resource title")
    url: HttpUrl = Field(..., description="Resource URL")
    provider: str = Field(..., description="Resource provider")
    type: str = Field(
        ..., description="Type of resource (course, book, tutorial, etc.)"
    )
    duration: str | None = Field(None, description="Estimated time to complete")
    level: str | None = Field(None, description="Difficulty level")
    cost: float = Field(0.0, description="Cost in USD")
    skills: list[str] = Field(
        default_factory=list, description="List of skills this resource helps with"
    )
    rating: float | None = Field(
        None, ge=0.0, le=5.0, description="Average rating (0-5)"
    )
    last_updated: datetime | None = Field(
        None, description="When the resource was last updated"
    )
