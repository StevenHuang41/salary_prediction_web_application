from pydantic import BaseModel, Field
from app.schemas.enums import GenderEnum, EducationLevelEnum


class SalaryFeatures(BaseModel):

    age: int = Field(
        default=28,
        ge=18,
        le=100,
        description="Age of the employee"
    )

    gender: GenderEnum = Field(
        default=GenderEnum.male,
        description="Gender of the employee"
    )

    education_level: EducationLevelEnum = Field(
        default=EducationLevelEnum.master,
        description="Education level"
    )

    job_title: str = Field(
        default="Data Scientist",
        min_length=2,
        max_length=64,
        description="Job title"
    )

    years_of_experience: float = Field(
        default=0,
        ge=0,
        le=82,
        description="Years of working experience"
    )


