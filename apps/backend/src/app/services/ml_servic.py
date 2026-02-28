import pandas as pd
from app.db.repositories.salary_repository import SalaryRepository
from app.core.config import settings
from my_package.data_predict import predict_salary


class SalaryMLService:

    def __init__(self, repo: SalaryRepository):
        self.repo = repo

    def predict(
        self,
        sample_df: pd.DataFrame,
        *,
        restart: bool = False,
    ) -> dict:

        df = self.repo.fetch_all()

        return predict_salary(
            sample_df=sample_df,
            df=df,
            store_file=str(settings.ARTIFACTS_DIR),
            restart=restart,
        )

