import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def box(salary: float, df: pd.DataFrame):
    """
    recv: salary and formData
    output: byte box image
    """

    percentile = (df.salary < salary).mean() * 100
    bins = np.arange(
        np.floor(df.salary.min()/10_000) * 10_000,
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
    return buf
