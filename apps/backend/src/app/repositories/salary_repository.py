from sqlalchemy.orm import Session
from sqlalchemy import select
import pandas as pd

from app.db.models.salary import SalaryRecord


class SalaryRepository:

    def get_all(self, db: Session):
        statement = select(SalaryRecord)

        return db.execute(statement).scalars().all()


    def get_dataframe(self, db: Session) -> pd.DataFrame:
        result = self.get_all(db)

        rows = [r.__dict__ for r in result]
        df = pd.DataFrame(rows)
        df = df.drop(columns=["_sa_instance_state"], errors="ignore")

        return df


    def insert(self, db: Session, record: dict):
        obj = SalaryRecord(**record)

        db.add(obj)
        db.commit()
        db.refresh(obj)

        return obj


    def delete_all(self, db: Session):
        db.query(SalaryRecord).delete()
        db.commit()


    def count(self, db: Session):
        return db.query(SalaryRecord).count()
