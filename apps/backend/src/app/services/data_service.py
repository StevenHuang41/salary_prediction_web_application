import io
import pandas as pd

from app.core.config import settings
from app.ml.data.cleaning import clean_data
from app.ml.plot import histogram, box


class DataService:
    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df if df is not None else pd.DataFrame()
        self._job_title_cache: list[str] | None = None
        self._is_load = False

    def load(self):
        if self._is_load:
            return

        if self.df.empty:
            self.df = pd.read_csv(settings.raw_data_file)

        self.df = clean_data(self.df, has_target_col=True)
        self._is_load = True

    def get_job_titles(self) -> list[str]:
        if self._job_title_cache is None:
            self._job_title_cache = sorted(
                self.df["job_title"].astype(str).unique()
            )
        return self._job_title_cache

    def plot_histogram(self, salary) -> io.BytesIO:
        return histogram(salary, self.df)

    def plot_box(self, salary) -> io.BytesIO:
        return box(salary, self.df)

    def add_record(self, new_recored: dict):
        new_df = pd.DataFrame([new_recored])
        self._job_title_cache = None
        self.df.loc[len(self.df)] = clean_data(new_df).iloc[0].to_dict()
        print('addre')
        return len(self.df)

    def reset(self) -> int:
        self.df = pd.DataFrame()
        self._job_title_cache = None
        self._is_load = False
        self.load()
        return len(self.df)


data_service = DataService()
