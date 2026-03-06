import io
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.data.cleaning import clean_data
from app.ml.plot import histogram, box
from app.repositories.salary_repository import SalaryRepository


class DataService:
    def __init__(self) -> None:
        self.repo = SalaryRepository()
        self.df: pd.DataFrame | None = None
        self._job_title_cache: list[str] | None = None


    def load(self, db: Session):
        df = self.repo.get_dataframe(db)

        if df is None or df.empty:
            self.df = pd.DataFrame()
            print("Database is empty")
            return

        self.df = clean_data(df, has_target_col=True)
        self._job_title_cache = None


    def get_job_titles(self, db: Session) -> list[str]:
        if self.df is None:
            self.load(db)

        if self._job_title_cache is None:
            self._job_title_cache = sorted(
                self.df["job_title"].astype(str).unique()
            )

        return self._job_title_cache

    def plot_histogram(self, db: Session, salary) -> io.BytesIO:
        if self.df is None:
            self.load(db)

        return histogram(salary, self.df)

    def plot_box(self, db: Session, salary) -> io.BytesIO:
        if self.df is None:
            self.load(db)

        return box(salary, self.df)

    def add_record(self, db: Session, new_recored: dict):
        new_df = pd.DataFrame([new_recored])
        cleaned = clean_data(new_df, has_target_col=True)

        obj = self.repo.insert(db, cleaned.iloc[0].to_dict())

        self.df = None
        self._job_title_cache = None

        return obj.id

    def reset(self, db: Session):
        self.repo.delete_all(db)

        self.df = None
        self._job_title_cache = None


    def seed(self, db: Session):
        if self.repo.count(db) > 0:
            db.close()
            return

        df = pd.read_csv(settings.raw_data_file)
        df = clean_data(df, has_target_col=True)

        records = df.to_dict(orient="records")

        for r in records:
            self.repo.insert(db, r)

        db.close()

    def reset_to_default(self, db: Session):
        self.reset(db)
        self.seed(db)
        

data_service = DataService()
