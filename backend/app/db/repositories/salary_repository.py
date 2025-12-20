import pandas as pd
from app.db.database import query_2_df
from app.core.config import settings


class SalaryRepository:
    def fetch_all(self) -> pd.DataFrame:
        return query_2_df(
            "SELECT * FROM salary",
            str(settings.DB_FILE)
        )

