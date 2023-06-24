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
    fit, res, *_ = np.polyfit(x, y, order, full=True)
    yhat = np.polyval(fit, x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((y-yhat)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    return fit, 1 - ssreg/sstot  # Highest power first


if __name__ == "__main__":
    mean, _ , velocity =  read_calibration('Group1')
    Velo = np.linspace(0, 20, 100)
    regression1,r1= polynomialfit(mean, velocity, 4)
    regression2,r2 = polynomialfit(velocity, mean, 4)
    plt.scatter(velocity, mean, color = 'r')
    plt.plot(Velo, np.polyval(regression2, Velo), color = 'black')
    plt.xlabel('$Flow velocity \quad [m/s]$')
    plt.ylabel('$Voltage \quad [V]$')
    plt.show()
    print(r1, r2, regression1, regression2)



