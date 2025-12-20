import app.db.state as state

def load_salary_df():
    if state.salary_df is None:
        raise RuntimeError("Database not initialized")
    return state.salary_df

