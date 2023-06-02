import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Calibration import read_calibration, polynomialfit


def read_measurements(folder: str, deg: int) -> list[(str, pd.DataFrame)]:
    Dataframelist = []
    if deg < 10:
        deg = f'0{deg}'

    deg = str(deg)

    for i in range(20):
        pos = 4 * i - 40
        if pos >= 0:
            sign = '+'
        else:
            sign = '-'

        pos_str = abs(pos)
        if pos_str < 10:
            pos_str = f"0{pos_str}"
        pos_str = str(pos_str)

        header = 22  # header and thereafter data on line 23
        sep = "\t"  # tab
        file = f'{folder}/Measurement_{sign}{pos_str}_{deg}'
        df = pd.read_table(file, sep=sep, header=header)
        # df['std_Voltage'] = df["Voltage"].std()
        # df['rms_Voltage'] = np.sqrt(df["Voltage"].mean() ** 2)

        Dataframelist.append((pos, df))

    return Dataframelist


def calc_velocity(Dataframelist: list[(float, pd.DataFrame)], regression: np.ndarray) -> (list[float],list[int]):
    lst = []
    rmsvelo = []
    pos = []
    for i in Dataframelist:
        i[1]['Velocity'] = np.polyval(regression, i[1]['Voltage'])
        rms_velo = np.sqrt(i[1]["Velocity"].mean() ** 2)
        # std_velo = i[2]['Velocity'].std()
        rmsvelo.append(rms_velo)
        pos.append(i[0])
    return rmsvelo, pos


def plot_rms(velocity: float, position: int, deg: int):
    plt.scatter(velocity, pos)
    plt.plot(velocity, pos)
    plt.title(f'NACA0012 at {deg} degrees')
    plt.xlabel('$V_{rms}$ $[m/s]$')
    plt.ylabel('$y$ $[mm]$')
    plt.show()


if __name__ == '__main__':
    folder = 'Group1'
    order = 4

    mean_voltage, _, velocity = read_calibration(folder)
    polynomial = polynomialfit(mean_voltage, velocity, order)

    for i in (0, 5, 15):
        deg = i
        measurements = read_measurements(folder, deg)
        rms_v, pos = calc_velocity(measurements, polynomial)
        plot_rms(rms_v, pos, deg)
