"""Canonical internal contract; this is not the frontend's /analyses response."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SourceKind = Literal["official", "synthetic"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    region_code: str = Field(pattern=r"^[0-9]{6,12}$")
    region_code_system: str = Field(min_length=1, max_length=80)
    industry_code: str = Field(min_length=1, max_length=80)
    period: str = Field(pattern=r"^\d{4}Q[1-4]$")
    available_at: date
    source_kind: SourceKind
    source_dataset: str = Field(min_length=1, max_length=200)
    sales_unit: Literal["KRW"] = "KRW"
    sales_amount: float | None = Field(default=None, ge=0)
    store_count: int | None = Field(default=None, ge=0, strict=True)
    opening_count: int | None = Field(default=None, ge=0, strict=True)
    closing_count: int | None = Field(default=None, ge=0, strict=True)
    resident_population: float | None = Field(default=None, ge=0)
    working_population: float | None = Field(default=None, ge=0)
    floating_population: float | None = Field(default=None, ge=0)
    area_km2: float | None = Field(default=None, gt=0)

    @field_validator("period")
    @classmethod
    def validate_period(cls, value: str) -> str:
        from pickplace_ai.periods import quarter_index

        quarter_index(value)
        return value

    @model_validator(mode="after")
    def validate_publication(self) -> "Observation":
        from pickplace_ai.periods import quarter_end

        if self.available_at < quarter_end(self.period):
            raise ValueError("available_at cannot precede the end of the observed quarter")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    region_code: str = Field(pattern=r"^[0-9]{6,12}$")
    region_code_system: str = Field(min_length=1, max_length=80)
    industry_code: str = Field(min_length=1, max_length=80)
    base_period: str = Field(pattern=r"^\d{4}Q[1-4]$")
    as_of: date
    source_kind: SourceKind
    history: list[Observation] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_history(self) -> "PredictionRequest":
        from pickplace_ai.periods import quarter_end, quarter_index

        base_index = quarter_index(self.base_period)
        if quarter_end(self.base_period) > self.as_of:
            raise ValueError("base_period must be complete as of the analysis date")
        periods: set[str] = set()
        for row in self.history:
            if (row.region_code, row.region_code_system, row.industry_code) != (
                self.region_code,
                self.region_code_system,
                self.industry_code,
            ):
                raise ValueError("All history rows must match the requested code system and scope")
            if row.source_kind != self.source_kind:
                raise ValueError("Cannot mix official and synthetic history")
            if row.period in periods:
                raise ValueError("Duplicate observation period")
            periods.add(row.period)
            if quarter_index(row.period) > base_index:
                raise ValueError("History cannot include periods after base_period")
            if row.available_at > self.as_of:
                raise ValueError("History includes data not available at as_of")
        if self.base_period not in periods:
            raise ValueError("history must include base_period")
        return self
