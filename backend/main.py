import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )

# from fastapi import FastAPI, Response
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# import os
# from pathlib import Path
#
# from app.schemas.salary import RowData, FullData, SalaryInput
# from app.api.router import router as api_router
#
# from my_package.data_cleansing import cleaning_data
# from my_package.data_extract_func import get_uniq_job_title
# from my_package.data_predict import predict_salary
# from my_package.data_visualization import (
#     salary_hist_image,
#     salary_box_image
# )
#
# from database.database import (
#     init_database,
#     create_index,
#     query_2_df,
#     insert_record
# )
#
# # Setup Local IP from .env.local
# def load_local_ip():
#     try :
#         with open('.env.local', 'r') as f:
#             for line in f:
#                 if 'http' in line:
#                     return line.strip()
#     except FileNotFoundError:
#         pass
#
#     return "http://127.0.0.1"
#
# LOCAL_IP = load_local_ip()
# LOCAL_FRONTEND = f"{LOCAL_IP}:3000"
#
# # Initialize FastAPI
# app = FastAPI()
# app.include_router(api_router)
#
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         f"{LOCAL_FRONTEND}",
#         "http://localhost:3000",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# print("CORS allow IP list:")
# print("  http://localhost:3000")
# print(f"  {LOCAL_FRONTEND}")
#
