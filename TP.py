import os
import pickle as pkl

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm, qr, solve
from scipy.linalg import expm

from rossler_map import RosslerMap

from TP5_ADL.statistics import time_correlation, get_frequencies


def lyapunov_exponent(traj, jacobian, max_it=1000, delta_t=1e-3):

    n = traj.shape[1]
    w = np.eye(n)
    rs = []
    chk = 0

    for i in range(max_it):
        jacob = jacobian(traj[i, :])
        # WARNING this is true for the jacobian of the continuous system!
        w_next = np.dot(expm(jacob * delta_t), w)
        # if delta_t is small you can use:
        #w_next = np.dot(np.eye(n)+jacob * delta_t,w)

        w_next, r_next = qr(w_next)

        # qr computation from numpy allows negative values in the diagonal
        # Next three lines to have only positive values in the diagonal
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)
        w = w_next
        if i//(max_it/100) > chk:
            print(i//(max_it/100))
            chk += 1

    return np.mean(np.log(rs), axis=0) / delta_t


def newton(f, jacob, x):
    # newton raphson method
    tol = 1
    while tol > 1e-5:
        # WARNING this is true for the jacobian of the continuous system!
        tol = x
        x = x-solve(jacob(x), f(v=x))
        tol = norm(tol-x)
    return x


if __name__ == '__main__':
    Niter = 100000
    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6,  0.02])
    if not os.path.isfile('generated_traj.dat'):

        traj, t = ROSSLER_MAP.full_traj(Niter, INIT)
        torch.save(torch.Tensor(traj), 'generated_traj.dat')

    else:
        traj = torch.load('generated_traj.dat')
        t = np.linspace(0, Niter * delta_t, Niter)
    traj_to_plot = traj[:10000]
    traj_to_plot = traj_to_plot[[False if i %
                                 8 else True for i in range(len(traj_to_plot))]]

    traj2 = torch.load('traj.dat')
    print('traj2', traj2.shape, 'traj', traj.shape)

    traj2_to_plot = traj2[:100000]

    traj2_to_plot = traj2_to_plot[[False if i %
                                   8 else True for i in range(len(traj2_to_plot))]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(traj_to_plot[:, 0], traj_to_plot[:, 1], traj_to_plot[:, 2], 'g')
    ax.plot(traj2_to_plot[:, 0], traj2_to_plot[:, 1], traj2_to_plot[:, 2], 'r')

    # fix_point = newton(ROSSLER_MAP.v_eq, ROSSLER_MAP.jacobian, INIT)
    with open('traj.pkl', 'wb') as fp:
        pkl.dump(traj, fp)

    plt.show()
    fig, ax = plt.subplots()
    m = min(len(traj), len(traj2))
    data = {
        'traj': traj[:m, 0],
        'traj2': traj2[:m, 0]
    }
    data = pd.DataFrame.from_dict(data)
    data.plot.kde(ax=ax, legend=False, title='Histogram: A vs. B')
    data.plot.hist(density=True, ax=ax)
    data.plot.hist(density=True, ax=ax)
    ax.set_ylabel('Probability')
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.show()

    time_correlation(10000, traj, traj2)

    k=20
    traj = traj[0:1000]
    traj2 = traj2[0:1000]
    time = range(len(traj))
    traj_freq = get_frequencies(traj)
    pred_freq = get_frequencies(traj2)
    traj_top = np.argsort(traj_freq)[-k:]
    pred_top = np.argsort(pred_freq)[-k:]

    plt.plot(time[:len(traj_freq) // 2], traj_freq[:len(traj_freq) // 2],label="real trajectory")
    plt.plot(time[:len(pred_freq) // 2], pred_freq[:len(pred_freq) // 2],label="predicted trajectory")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.legend()
    plt.title("Frequencies for the time range [0,1000] ")
    plt.show()