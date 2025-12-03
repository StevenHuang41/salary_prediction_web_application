from .data_cleansing import cleaning_data, clean_directory
from .data_spliting import spliting_data
from .data_preprocessing import preprocess_data
from .data_training import model_select

import os
import shutil
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from keras.models import load_model


def predict_salary(
    sample_df: pd.DataFrame,
    df: pd.DataFrame,
    store_file: str,
    restart: bool = False,
) -> dict:
    ## spliting
    X_train, X_test, y_train, y_test = spliting_data(df, random_state=23)

    best_performance_dir = store_file

    if restart:
        try :
            # shutil.rmtree(f'{store_file}')
            clean_directory(store_file)
        except FileNotFoundError:
            pass

    try :
        best_performance_dir_list = os.listdir(best_performance_dir)
    except Exception:
        best_performance_dir_list = []

    if best_performance_dir_list == []:
        # missing best performance, process model selection
        model_name, model = model_select(
            X_train, y_train, X_test, y_test, best_performance_dir
        )

    else :
        model_name = best_performance_dir_list[0].split('.')[0]

        model_postfix = '.keras' if 'NN' in model_name else '.joblib'
        model_path = os.path.join(best_performance_dir,
                                  f"{model_name}{model_postfix}")
        ## load model
        model = load_model(model_path) \
                if 'NN' in model_name \
                else joblib.load(model_path)

    with open(f'{best_performance_dir}/{model_name}.json', 'r') as f:
        model_params = json.load(f)

    ## preprocessing
    use_poly = True if 'poly' in model_name else False
    _, example_df_ = preprocess_data(X_train, y_train, sample_df,
                                     use_polynomial=use_poly)

    model_name_trim = model_name.split('_')[0]

    return {
        "model_name": model_name_trim,
        "use_polynomial": use_poly,
        "value": float(model.predict(example_df_)[0]),
        "num_train_dataset": X_train.shape[0],
        "num_test_dataset": X_test.shape[0],
        "params": model_params,
    }

if __name__ == "__main__":

    from data_cleansing import cleaning_data

    p_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    FILE_PATH = os.path.join(p_dir, 'database/Salary_Data.csv')
    df = pd.read_csv(FILE_PATH)
    df = cleaning_data(df)

    store_file = 'best'
    sample_df = pd.DataFrame([{
        "age": 20,
        "gender": "female",
        "education_level": "Bachelor",
        "job_title": "Data Engineer",
        "years_of_experience": 1,
    }])

    # test 1
    # print(predict_salary(sample_df, df, store_file))

    # test 2
    # print(predict_salary(sample_df, df, store_file, restart=True))


    shutil.rmtree(store_file)