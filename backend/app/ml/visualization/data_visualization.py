import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import floor
from typing import Literal
# from .data_cleansing import cleaning_data

def show_plot(
    df: pd.DataFrame,
    col: str,
    *,
    group_method: Literal['mean', 'median', 'mode'] = 'mean'
):
    ## create images directory 
    images_dir = os.path.join(os.getcwd(), 'images')
    os.makedirs(images_dir, exist_ok=True)

    title_ = col

    ## show distribution
    plt.figure(figsize=(10, 5), dpi=100)
    if col == 'job_title':
        plt.xticks(rotation=90)

        # group jobs that less than 45
        threshold = 45
        temp_df = df[[col, 'salary']]
        job_counts = temp_df[col].value_counts()
        valid_jobs = job_counts[job_counts > threshold].index

        # change jobs lower than threshold name to 'Other'
        temp_df.loc[:, col] = (
            temp_df[col]
            .where(temp_df[col].isin(valid_jobs), other='Other')
        )

        plot_df = (
            temp_df
            .groupby([col], observed=True).salary
            .agg(count='count',
                 mean='mean',
                 median=lambda x: x.median(),
                 mode=lambda x: x.mode().mean(),)
        )

        # place row: Other at the end
        plot_df = pd.concat([
            plot_df.drop('Other'),
            plot_df.loc[['Other'], :]
        ])

        # plot barplot
        bars = sns.barplot(data=plot_df['count'],
                           color=(0.4, 0.9, 0.9),
                           edgecolor='black',
                           saturation=1,
                           alpha=0.8,
                           width=1)
        bars.margins(x=0.05)

    else :
        # plot histgram
        bars = sns.histplot(data=df,
                            x=col,
                            color=(0.4, 0.9, 0.9),
                            alpha=0.8)

    ## indicate number on bar
    for bar in bars.patches:
        height = bar.get_height()
        bars.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            int(height),
            ha='center',
            va='bottom',
            fontsize=6,
        )

    if '_' not in title_:
        title_ = title_[0].upper() + title_[1:]
    else :
        title_ = " ".join([n_split.capitalize()
                           if n_split != 'of'
                           else n_split
                           for n_split in title_.split('_')])

    plt.xlabel('')
    plt.ylabel('Count')
    plt.title(f"{title_}")
    plt.tight_layout()

    ## save image
    plt.savefig(os.path.join(images_dir, f'{col}_histogram.png'),
                bbox_inches='tight')

    ## plot image
    plt.show()

    ################## Group Mean Salary ##################

    if col == 'salary':
        return

    ## get feature mean|median|mode with target feature
    group_df = (
        df
        .groupby([col], observed=True)
        .salary
        .agg(mean='mean',
             median=lambda x: x.median(),
             mode=lambda x: x.mode().mean())
    )

    plt.figure(figsize=(12, 5), dpi=100)
    if col == 'job_title':
        plt.xticks(rotation=90)
        group_df = plot_df

    plt.bar(group_df.index,
            group_df[group_method],
            width=1,
            color=(0.9, 0.4, 0.9),
            edgecolor='black',
            alpha=0.8)

    for x, y in  zip(group_df.index, group_df[group_method]):
        plt.text(x, y, str(int(y)), ha='center', va='bottom', fontsize=6)

    plt.xlabel('')
    plt.ylabel(f"Group {group_method} salary")
    plt.title(title_)
    plt.tight_layout()

    ## save image
    fig_name = f'{col}_group{group_method.capitalize()}_salary.png'
    plt.savefig(os.path.join(images_dir, fig_name), bbox_inches='tight')

    ## plot image
    plt.show()


def show_heatmap(X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 use_poly: bool = False) -> None:
    ## create images directory 
    images_dir = os.path.join(os.getcwd(), 'images')
    os.makedirs(images_dir, exist_ok=True)

    X_train['salary'] = y_train

    annot_size = 10
    if use_poly:
        annot_size = 7

    plt.figure(figsize=(10, 10))
    sns.heatmap(X_train.corr(),
                annot=True,
                cmap='coolwarm',
                annot_kws={'size': annot_size})

    plt.tight_layout()
    

    ## save fig
    fig_name = 'features_heatmap_poly.png' \
                if use_poly \
                else 'features_heatmap.png'

    plt.savefig(os.path.join(images_dir, fig_name), bbox_inches='tight')

    ## show fig
    plt.show()

def salary_hist_image(salary: float, df: pd.DataFrame):
    """
    recv: salary and pd.DataFrame
    output: byte hist image
    """

    percentile = (df.salary < salary).mean() * 100
    bins = np.arange(
        floor(df.salary.min()/10_000) * 10_000,
        df.salary.max() + 10_000,
        20_000
    )

    sns.set_theme('paper')
    plt.figure(figsize=(10, 6), dpi=300)

    sns.histplot(data=df, x='salary', bins=bins, kde=True)
    sns.kdeplot(data=df.salary, label='KDE')

    plt.axvline(salary, color='lightgreen', linestyle='-',
                label=f'predict salary: {salary:.2f}',
                linewidth=3)

    plt.plot([],[],' ', label=f"percentile: {percentile:.2f}%")

    plt.xlabel('', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title('Salary Histogram with KDE Line', fontsize=30)
    plt.legend(fontsize=15)
    plt.xticks(bins, [f'{v//1000:.0f}k' for v in bins], fontsize=15)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def salary_box_image(salary: float, df: pd.DataFrame):
    """
    recv: salary and formData
    output: byte box image
    """

    percentile = (df.salary < salary).mean() * 100
    bins = np.arange(
        floor(df.salary.min()/10_000) * 10_000,
        df.salary.max() + 10_000,
        20_000
    )

    sns.set_theme('paper')
    plt.figure(figsize=(10, 6), dpi=300)

    sns.boxplot(data=df.salary, orient='h', color='skyblue')
    plt.axvline(salary, color='lightgreen', linestyle='-',
                linewidth=3,
                label=f"predict salary: {salary:.2f}")
    plt.plot([],[],' ', label=f"percentile: {percentile:.2f}%")

    plt.xlabel('')
    plt.title('Salary Box Plot', fontsize=30)
    plt.legend(fontsize=15)
    plt.xticks(bins, [f'{v//1000:.0f}k' for v in bins], fontsize=15)
    plt.tight_layout()
    # plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    import shutil

    ### load csv 
    ## dataFrame
    # from data_cleansing import cleaning_data
    # from data_spliting import spliting_data
    # from data_preprocessing import preprocess_data

    # FILE_NAME = "../database/Salary_Data.csv"
    # df = pd.read_csv(FILE_NAME, delimiter=',')
    # df = cleaning_data(df, has_target_columns=True)

    ## sql
    import sys
    from data_cleansing import cleaning_data
    p_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(p_dir)
    from database.database import query_2_df

    df = query_2_df("select * from salary;")
    df = cleaning_data(df)
    
    images_dir = os.path.join(os.getcwd(), 'images')
    os.makedirs(images_dir, exist_ok=True)

    # # test 1
    # for col in df.columns:
    #     show_plot(df, col, group_method='median')


    # X_train, X_test, y_train, y_test = spliting_data(df)
    ## test 2
    # X_train_, X_test_ = preprocess_data(
    #     X_train, y_train, X_test, use_polynomial=True
    # )
    # show_heatmap(X_train_, y_train, use_poly=True)

    ## test 3
    # X_train_, X_test_ = preprocess_data(
    #     X_train, y_train, X_test, use_polynomial=False
    # )
    # show_heatmap(X_train_, y_train, use_poly=False)

    # test 4
    # salary_hist_image(salary=100000, df=df)

    # test 5
    salary_box_image(100000, df)

    shutil.rmtree(images_dir)
    pass