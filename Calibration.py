import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_calibration(directory: str) -> (np.ndarray,np.ndarray,np.ndarray):
    std_voltage = []
    mean_voltage = []
    approximate_velocity = np.arange(0, 22, 2)

    header = 22  # header and thereafter data on line 23
    sep = "\t"  # tab

    for i in range(0,220,20):
        file = f'{directory}/Calibration_{i:03}'
        df = pd.read_table(file, sep=sep, header=header)
        std_voltage.append(df["Voltage"].std())
        mean_voltage.append(df["Voltage"].mean())

    mean_voltage = np.array(mean_voltage)
    std_voltage = np.array(std_voltage)


    return mean_voltage, std_voltage, approximate_velocity

def polynomialfit(x: np.ndarray ,y: np.ndarray, order: int, std: np.ndarray | None = None) -> np.ndarray:
    """Currently does not do a weighted leastsquares which could be an improvement as we know the standard deviation
    of the datapoints. however numpy implements polyfit with weights wrongly so rather do least squares when implementing"""
    coefficients = np.polyfit(x, y, order)
    return coefficients  # Highest power first


if __name__ == "__main__":
    mean, _ , velocity =  read_calibration('Group1')
    regression = polynomialfit(mean, velocity, 4)
    plt.scatter(mean, velocity)
    plt.plot(mean, np.polyval(regression, mean))
    plt.show()



