import pickle as pkl

import torch
import torch.nn as nn
from tqdm import tqdm


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.l1 = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(10, 3)

    def forward(self, inputs):
        o1 = self.relu(self.l1(inputs))
        return self.l2(o1)


class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t  # if discrete model your delta_t
        # if continuous model chose one <=1e-2
        self.nb_steps = 5300//self.delta_t

        self.rosler_nn = Model()
        self.rosler_nn.load_state_dict(torch.load('model.pt'))

    def full_traj(self, initial_condition=[-5.75, -1.6,  0.02]):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        y = []
        cur_pos = torch.Tensor(initial_condition)
        with torch.no_grad():
            for _ in tqdm(range(int(self.nb_steps)), total=self.nb_steps):
                cur_pos = self.rosler_nn(cur_pos.view(1, -1)).flatten()
                y.append(cur_pos)

        return torch.cat(y, dim=-1).view(-1, 3)

    def save_traj(self, y):
        print('y', y.shape)
        torch.save(y, 'traj.dat')


if __name__ == '__main__':

    ROSSLER = Rossler_model(1e-2)

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)
