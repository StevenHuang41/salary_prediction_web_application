from app.db.repositories.salary_repository import SalaryRepository

_repo = SalaryRepository()

def get_salary_df():
    return _repo.fetch_all()

