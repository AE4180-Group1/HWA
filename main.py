import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_measurments(folder: str, deg: int) -> list[(str, str, pd.DataFrame)]:
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
        pos = abs(pos)

        if pos < 10:
            pos = f"0{pos}"
        str(pos)

        header = 22  # header and thereafter data on line 23
        sep = "\t"  # tab
        file = f'{folder}/Measurement_{sign}{pos}_{deg}'
        df = pd.read_table(file, sep=sep, header=header)
        df['std_Voltage'] = df["Voltage"].std()
        df['rms_Voltage'] = np.sqrt(df["Voltage"].mean() ** 2)

        Dataframelist.append((deg, pos, df))

    return Dataframelist


def calc_velocity(Dataframelist: list[(str, str, pd.DataFrame)], regression: np.ndarray) -> None:
    for i in Dataframelist:
        i[2]['Velocity'] = np.polyval(regression, i[2]['Voltage'])
        i[2]['rms_velo'] = np.sqrt(i[2]["Velocity"].mean() ** 2)
        i[2]['std_velo'] = i[2]['Velocity'].std()


if __name__ == '__main__':
    rms 
    print(read_measurments("Group1", 0))
