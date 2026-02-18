from app.db.repositories.salary_repository import SalaryRepository

def get_salary_repository() -> SalaryRepository:
    return SalaryRepository()

