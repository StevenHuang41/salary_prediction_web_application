from pathlib import Path
import sqlite3
import pandas as pd

from app.core.config import CSV_FILE

def init_database(
    db: str,
    *,
    seed_if_empty: bool = True,
    chunksize: int = 200_000,
) -> None:

    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema = """
        age INTEGER,
        gender TEXT,
        education_level TEXT,
        job_title TEXT,
        years_of_experience REAL,
        salary REAL
    """

    """
    4 age INTEGER,
    6 gender TEXT,
    12 education_level TEXT,
    25 job_title TEXT,
    8 years_of_experience REAL,
    8 salary REAL
    64B * 200_000 = 12.8MB
    """

    with sqlite3.connect(db) as conn:
        c = conn.cursor()

        c.execute(f"""
            CREATE TABLE IF NOT EXISTS salary ({schema});
        """)
        conn.commit()

        if not seed_if_empty:
            return

        # check if table already has data
        c.execute("SELECT COUNT(*) FROM salary;")
        count = c.fetchone()[0]
        if count > 0:
            # tabe has data
            return


        # import csv to db
        if not CSV_FILE.exists():
            raise FileNotFoundError(f"Missing csv file {CSV_FILE}")

        from my_package.data_cleansing import cleaning_data

        for chunk in pd.read_csv(CSV_FILE, chunksize=chunksize):
            chunk = cleaning_data(chunk, has_target_columns=True)
            chunk.to_sql('salary', conn, if_exists='append', index=False)

        conn.commit()


def create_index(
    col: str,
    idx_name: str,
    table: str='salary',
    db: str='salary_prediction.db',
):
    # db_path = os.path.join(os.path.dirname(__file__), db)
    with sqlite3.connect(db) as conn:
        c = conn.cursor()

        c.execute(f"""
            drop index if exists {idx_name};
        """)

        c.execute(f"""
            create index {idx_name} on {table}({col});
        """)

        conn.commit()

def create_view(
    view_name: str,
    query: str,
    db: str='salary_prediction.db',
):
    # db_path = os.path.join(os.path.dirname(__file__), db)
    with sqlite3.connect(db) as conn:
        c = conn.cursor()

        c.execute(f"""
            DROP VIEW IF EXISTS {view_name};
        """)
        c.execute(f"""
            create view {view_name} as
            {query}
        """)

        conn.commit()


def query_show_r(
    query: str,
    db: str='salary_prediction.db'
):
    # db_path = os.path.join(os.path.dirname(__file__), db)
    with sqlite3.connect(db) as conn:
        c = conn.cursor()

        c.execute(f"{query}")
        for row in c:
            print(row)

        conn.commit()


def query_2_df(query: str, db: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(query, conn)

    return df

def insert_record(record: dict, table: str, db: str):
    with sqlite3.connect(db) as conn:
        c = conn.cursor()

        c.execute(f"PRAGMA table_info({table})")
        schema_info = c.fetchall()
        table_column = [col[1] for col in schema_info]

        # check record keys
        record_keys_set = set(record.keys())
        if record_keys_set != set(table_column):
            raise AssertionError("Record keys do not match table columns.")

        import sys
        from my_package.data_cleansing import cleaning_data

        record_df = pd.DataFrame([record])
        record_df = cleaning_data(record_df)
        record_df.to_sql('salary', conn, if_exists='append', index=False)

        conn.commit()

def delete_record(rowid, db: str) -> None:
    with sqlite3.connect(db) as conn:
        c = conn.cursor()

        c.execute("delete from salary where rowid = (?)", (str(rowid),))
        conn.commit()
