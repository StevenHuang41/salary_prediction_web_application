from decimal import Decimal
from sqlalchemy import String, Integer, Float, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class SalaryRecord(Base):
    __tablename__ = "salary_records"

    id: Mapped[int] = mapped_column(primary_key=True)

    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(8))
    education_level: Mapped[str] = mapped_column(String(16))
    job_title: Mapped[str] = mapped_column(String(64))

    job_seniority: Mapped[str] = mapped_column(String(32))
    job_group: Mapped[str] = mapped_column(String(32))
    job_role: Mapped[str] = mapped_column(String(32))

    years_of_experience: Mapped[float] = mapped_column(Float)
    salary: Mapped[Decimal] = mapped_column(Numeric(10, 2))
