import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt


def read(directory: str, file: str) -> (np.ndarray, np.ndarray, np.ndarray):
    approximate_velocity = np.arange(0, 22, 2)
    header = 22  # header and thereafter data on line 23
    sep = "\t"  # tab
    file = f'{directory}/{file}'
    df = pd.read_table(file, sep=sep, header=header)
    return df


def autocorr(df: pd.Series):
    lag = [*range(0, len(df)/2)]
    autocor = []
    for i in lag:
        corri = df.autocorr(lag=i)
        autocor.append(corri)

    plt.plot(lag, autocor, color='black')
    plt.ylabel("Auto correlation coefficient")
    plt.xlabel("Lag in samples")
    plt.axhline(y=0.1, color='r', linestyle=':')
    plt.axhline(y=-0.1, color='r', linestyle=':')
    plt.plot()
    plt.show()
    return autocor


def plot(df: pd.Series):
    pd.plotting.autocorrelation_plot(df)
    plt.show()


if __name__ == "__main__":
    df = read("Group1", "CorrelationTest")
    # plot(df["Voltage"])
    autocorr(df["Voltage"])
