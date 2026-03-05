from pydantic import BaseModel, Field
from app.schemas.enums import GenderEnum, EducationLevelEnum


class SalaryFeatures(BaseModel):

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Age of the employee"
    )

    gender: GenderEnum = Field(
        ...,
        description="Gender of the employee"
    )

    education_level: EducationLevelEnum = Field(
        ...,
        description="Education level"
    )

    job_title: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Job title"
    )

    years_of_experience: float = Field(
        ...,
        ge=0,
        le=82,
        description="Years of working experience"
    )

    
