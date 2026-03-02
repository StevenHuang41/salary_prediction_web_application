from pydantic import BaseModel


class PlotRequest(BaseModel):
    salary: float
