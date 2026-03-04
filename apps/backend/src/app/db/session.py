from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

database_url = settings.database_url
if database_url is None:
    raise ValueError("DATABASE_URL is not set")

engine = create_engine(
    database_url,
    echo=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
    bind=engine,
)
