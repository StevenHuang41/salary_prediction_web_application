from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/

DATABASE_DIR = BASE_DIR / "database"

DB_FILE = DATABASE_DIR / "salary_prediction.db"

MODEL_DIR = BASE_DIR / "best_performance"

CSV_FILE = DATABASE_DIR / "salary_data.csv"

