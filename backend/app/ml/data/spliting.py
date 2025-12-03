from sklearn.model_selection import train_test_split
import pandas as pd

def spliting_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return train_test_split(X, y,
                            test_size=test_size,
                            shuffle=shuffle,
                            random_state=random_state)


if __name__ == "__main__":
    from data_cleansing import cleaning_data

    # load csv 
    FILE_NAME = "../Salary_Data.csv"
    df = pd.read_csv(FILE_NAME, delimiter=',')
    
    df = cleaning_data(df, has_target_columns=True)

    # test
    X_train, X_test, y_train, y_test = spliting_data(df)
    print(X_train.info())
    print(y_train.info())
    print(X_test.info())
    print(y_test.info())