import torch
import matplotlib.pyplot as plt
import numpy as np


def time_correlation(time_range, traj, y_pred):
    t = 0
    counter = 0
    s = traj.size()[0]
    size = int((s - time_range) / time_range)+1
    traj_correlations = torch.zeros(size)
    pred_correlations = torch.zeros(size)
    for T in range(time_range, s, time_range):
        traj_correlations[counter] = correlation(traj, t, T)
        pred_correlations[counter] = correlation(y_pred, t, T)
        t = T
        counter += 1
    plt.plot(traj_correlations.numpy(),label="Real trajectory")
    plt.plot(pred_correlations.detach().numpy(),label="Predicted trajectory")
    plt.legend()
    plt.xlabel("time range")
    plt.ylabel("Time correlation")
    plt.title("Time correlation for T = {}".format(time_range))
    plt.show()


def correlation(w, t, T):
    mu = torch.mean(w)
    return torch.sum((w[t] - mu) * (w[T] - mu)) / (
            torch.sqrt(sum((w[t] - mu) ** 2)) * torch.sqrt(torch.sum((w[T] - mu) ** 2)))


def get_frequencies(data):
    if type(data) is not np.ndarray:
        data = data.numpy()
    data = (data - np.mean(data)) / np.std(data)
    fftdata = np.fft.fft(data)
    fftdatafreq = np.zeros((len(data)))
    for i in range(len(fftdata)):
        fftdatafreq[i] = abs(fftdata[i].real)[2]
    return fftdatafreq


