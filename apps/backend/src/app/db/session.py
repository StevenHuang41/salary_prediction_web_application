from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

db_url = settings.db_url
if db_url is None:
    raise ValueError("DATABASE_URL is not set")

engine = create_engine(
    db_url,
    echo=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
    bind=engine,
)
