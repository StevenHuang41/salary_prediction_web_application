import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


def histogram(salary: float, df: pd.DataFrame) -> io.BytesIO:
    """
    recv: salary and pd.DataFrame
    output: byte hist image
    """

    percentile = (df.salary < salary).mean() * 100
    bins = np.arange(
        np.floor(df.salary.min()/10_000) * 10_000,
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

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
