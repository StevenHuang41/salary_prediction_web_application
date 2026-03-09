from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

if settings.db_url is None:
    raise ValueError("DATABASE_URL is not set")

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.db_url,
            echo=True,
            pool_pre_ping=True,
        )

    return _engine

SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
    bind=get_engine(),
)
