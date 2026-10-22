"""Request and response schemas for abalone predictions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputData(BaseModel):
    """Measurements for one abalone, using the training data's column names."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    Sex: Literal["M", "F", "I"] = Field(description="Male, female, or infant.")
    Length: float = Field(gt=0, description="Shell length.")
    Diameter: float = Field(gt=0, description="Shell diameter.")
    Height: float = Field(gt=0, le=0.4, description="Shell height.")
    Whole_weight: float = Field(gt=0, alias="Whole weight", description="Whole weight.")
    Shucked_weight: float = Field(
        gt=0, alias="Shucked weight", description="Shucked meat weight."
    )
    Viscera_weight: float = Field(
        gt=0, alias="Viscera weight", description="Viscera weight."
    )
    Shell_weight: float = Field(gt=0, alias="Shell weight", description="Shell weight.")


class PredictionOut(BaseModel):
    """The model's predicted ring count, used as an estimate of age."""

    age: float = Field(description="Predicted number of rings.")
