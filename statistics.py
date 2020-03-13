import torch
import matplotlib.pyplot as plt


def time_correlation(time_range, traj, y_pred):
    t = 0
    counter = 0
    N = traj.size()[0]
    size = int((N - time_range) / time_range)
    traj_correlations = torch.zeros(size)
    pred_correlations = torch.zeros(size)
    for T in range(time_range, N, time_range):
        traj_correlations[counter] = correlation(traj, t, T)
        pred_correlations[counter] = correlation(y_pred, t, T)
        t = T
        counter += 1
    plt.plot(traj_correlations.numpy())
    plt.plot(pred_correlations.detach().numpy())
    plt.show()


def correlation(w, t, T):
    mu = torch.mean(w)
    return torch.sum((w[t] - mu) * (w[T] - mu)) / (
            torch.sqrt(sum((w[t] - mu) ** 2)) * torch.sqrt(torch.sum((w[T] - mu) ** 2)))
