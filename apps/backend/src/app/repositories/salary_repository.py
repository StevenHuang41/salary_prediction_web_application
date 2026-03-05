import pandas as pd
from sqlalchemy.orm import Session
from app.db.models import SalaryRecord


class SalaryRepository:

    def get_all(self, db: Session) -> pd.DataFrame:
        query = db.query(SalaryRecord)

        df = pd.read_sql(
            query.statement,
            db.bind
        )

        return df

    def insert(self, db: Session, record: dict):

        obj = SalaryRecord(**record)

        db.add(obj)
        db.commit()

        return obj.id

    def reset(self, db: Session):

        db.query(SalaryRecord).delete()
        db.commit()
