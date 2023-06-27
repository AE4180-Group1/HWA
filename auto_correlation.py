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


def plot_autocorr(df: pd.Series) -> (list, np.ndarray):
    data = df.to_numpy()
    fftautocor, size = fftautocorr(data)
    autocor = normalcorr(data)
    lag = np.array([*range(0, size)]) * 0.0001
    # autocor = []
    # for i in lag:
    #     corri = df.autocorr(lag=i)
    #     autocor.append(corri)

    plt.plot(lag, autocor, color='black', label = 'normal')
    plt.plot(lag, fftautocor, color='blue', label = 'fft')
    plt.ylabel("Auto correlation coefficient [-]")
    plt.xlabel("Lag [s]")
    plt.axhline(y=0.1, color='r', linestyle=':')
    plt.axhline(y=-0.1, color='r', linestyle=':')
    plt.legend()
    plt.show()
    return autocor, fftautocor
def fftautocorr(data: np.ndarray) -> (np.ndarray, int):
    # size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int') # Nearest size with power of 2
    size = len(data) * 2
    var = np.var(data)  # Variance
    ndata = data - np.mean(data) # Normalized data
    ndata = np.pad(ndata, (0, len(data)), "constant") # padding normalised data with zeros
    fft = np.fft.fft(ndata, len(data))  # Compute the FFT
    pwr = np.abs(fft) ** 2 # Get the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(data) # Calculate the autocorrelation from inverse FFT of the power spectrum
    return acorr[0:len(data)], len(data)

def normalcorr(data: np.ndarray) -> np.ndarray:
    ndata = data - np.mean(data)  # Normalized data
    ndatapadded = np.pad(ndata, (0, len(data)), "constant")  # padding normalised data with zeros
    corr = np.correlate(ndata,ndatapadded,"valid")
    return corr[-1:0:-1]/corr[-1]

def integral_time(correlation: np.ndarray):
    lag = np.arange(0,len(correlation))
    time = lag * 0.0001 # sampling interval in correlation test
    return sc.integrate.simpson(correlation, time)


# def plot(df: pd.Series):
#     pd.plotting.autocorrelation_plot(df)
#     plt.show()


if __name__ == "__main__":
    df = read("Group1", "CorrelationTest")
    autocor, fftautocor = plot_autocorr(df["Voltage"])
    # np.savetxt("normalpadded.csv", autocor, delimiter=",")
    # np.savetxt("fftpadded.csv", fftautocor, delimiter=",")
    # normal = np.genfromtxt("normal2.csv", delimiter= ",")
    # normal[-1] = 1.
    # fft = np.genfromtxt("fft.csv", delimiter= ",")
    # normalfull = np.genfromtxt("normal.csv", delimiter= ",")
    # normalfull[-1] = 1.
    # fftfull = np.genfromtxt("fft.csv", delimiter= ",")
    normalp= np.genfromtxt("normalpadded.csv", delimiter= ",")
    # normalfull[-1] = 1.
    fftp = np.genfromtxt("fftpadded.csv", delimiter= ",")
    print(integral_time(normalp[0:25000]), integral_time(fftp[0:25000]))
    print(integral_time(normalp), integral_time(fftp))

