from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os

from my_package.data_cleansing import cleaning_data
from my_package.data_extract_func import get_uniq_job_title
from my_package.data_predict import predict_salary
from my_package.data_visualization import (
    salary_hist_image,
    salary_box_image
)

from database.database import (
    init_database,
    create_index,
    query_2_df,
    insert_record
)

# Setup Local IP from .env.local
def load_local_ip():
    try :
        with open('.env.local', 'r') as f:
            for line in f:
                if 'http' in line:
                    return line.strip()
    except FileNotFoundError:
        pass
    
    return "http://127.0.0.1"

LOCAL_IP = load_local_ip()
LOCAL_FRONTEND = f"{LOCAL_IP}:3000"

# Initialize FastAPI 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"{LOCAL_FRONTEND}",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("CORS allow IP list:")
print("  http://localhost:3000")
print(f"  {LOCAL_FRONTEND}")

# Database Initialization
DB_DIR = os.path.join(os.getcwd(), 'database')
DB_FILE = os.path.join(DB_DIR, 'salary_prediction.db')

init_database(DB_FILE)
create_index('job_title', 'idx_job_title', db=DB_FILE)
create_index('education_level', 'idx_education_level', db=DB_FILE)
create_index('salary', 'idx_salary', db=DB_FILE)

def load_salary_df():
    return query_2_df("select * from salary", DB_FILE)

# Data Models
class RowData(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

class FullData(RowData):
    salary: float

class SalaryInput(BaseModel):
    salary: float

## dataFrame needs cleansing
# df = pd.read_csv("database/Salary_Data.csv")
# df = cleaning_data(df, has_target_columns=True)
# print(df)



## file path
CURRENT_FILE = os.getcwd()
STORE_MODEL_FILE = os.path.join(CURRENT_FILE, "best_performance")

@app.get('/api/get_uniq_job_title')
async def api_get_uniq_job_title():
    df = load_salary_df()
    result = get_uniq_job_title(df)

    return {'value': result}


@app.post("/api/predict")
async def api_predict_salary(data: RowData):
    df = load_salary_df()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    result = predict_salary(input_df, df, STORE_MODEL_FILE)
    """
    result:
        "model_name": model_name_trim,
        "use_polynomial": use_poly,
        "value": float(model.predict(example_df_)[0]),
        "num_train_dataset": X_train.shape[0],
        "num_test_dataset": X_test.shape[0],
        "params": model_params,
    """
    return result


@app.post("/api/retrain_model")
async def api_retrain(data: RowData):
    df = load_salary_df()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    result = predict_salary(
        input_df,
        df,
        STORE_MODEL_FILE,
        restart=True,
    )

    return {
        'status': 'success',
        'message': 'Retrain model successfully.',
        'result': result,
    }

@app.post("/api/reset_model")
async def api_reset_model():
    init_database(DB_FILE)
    create_index('job_title', 'idx_job_title', db=DB_FILE)
    create_index('education_level', 'idx_education_level', db=DB_FILE)
    create_index('salary', 'idx_salary', db=DB_FILE)

    return {
        'status': 'success',
        'message': 'Reset the database.',
    }

@app.post("/api/add_data")
async def api_add_record(data: FullData):
    input_df = cleaning_data(
        pd.DataFrame([data.model_dump()]),
        has_target_columns=True,
    )
    record = input_df.to_dict(orient="records") 

    if record == []:
        return {'status': 'fail',
                'message': 'Input data does not add into database.'}

    data_dict = record[0]
    ## insert data into database
    insert_record(data_dict, 'salary', DB_FILE)
    # print(data_dict)
    return {
        'status': 'success',
        'message': 'Input data stored in database.'
    }



@app.post("/api/salary_avxline_plot")
async def api_salary_hist(data: SalaryInput):
    df = load_salary_df()
    image_byte = salary_hist_image(data.salary, df)

    return Response(content=image_byte, media_type="image/png")

@app.post("/api/salary_boxplot")
async def api_salary_boxplot(data: SalaryInput):
    df = load_salary_df()
    image_byte = salary_box_image(data.salary, df)

    return Response(content=image_byte, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


## TODO: learn how to use pydantic and typing and use in my_package
## TODO: setup splite database

"""
{
  "age": 30,
  "gender": "male",
  "education_level": "master",
  "job_title": "Data Scientist",
  "years_of_experience": 3
}
"""
