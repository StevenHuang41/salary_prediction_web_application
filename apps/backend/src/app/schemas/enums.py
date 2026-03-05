from enum import Enum


class GenderEnum(str, Enum):
    male = "male"
    female = "female"
    other = "other"


class EducationLevelEnum(str, Enum):
    high_school = "High School"
    bachelor = "Bachelor"
    master = "Master"
    phd = "PhD"


