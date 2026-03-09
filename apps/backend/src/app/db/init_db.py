from app.db.base import Base
from app.db.session import get_engine
from app.db.models import SalaryRecord


def init_db():
    Base.metadata.create_all(bind=get_engine())

