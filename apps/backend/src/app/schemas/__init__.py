from .prediction import PredictRequest, PredictResponse
from .records import AddRecordRequest, AddRecordResponse
from .plot import PlotRequest
from .features import SalaryFeatures
from .enums import GenderEnum, EducationLevelEnum

__all__ = [
    "PredictRequest",
    "PredictResponse",
    "AddRecordRequest",
    "AddRecordResponse",
    "PlotRequest",
    "SalaryFeatures",
    "GenderEnum",
    "EducationLevelEnum",
]
