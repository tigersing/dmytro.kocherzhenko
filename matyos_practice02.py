import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Matyos_function(nn.Module):
    def __init__(self, x0, y0):
        super(Matyos_function, self).__init__()
        self.x = nn.Parameter(x0)
        self.y = nn.Parameter(y0)

    def forward(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        loss = torch.exp(x) / (x ** 2 + y ** 2 + 1)
        return loss

    def string(self):
        return f'x = {self.x.item()} y = {self.y.item()}'

model = Matyos_function(torch.tensor(4.0).to(device), torch.tensor(4.0).to(device))

x = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, x)

Z = np.empty((50, 50), dtype=np.float32)
i, j = 0, 0
for x_val in np.linspace(-10, 10, 50):
    for y_val in np.linspace(-10, 10, 50):
        Z[i, j] = model.forward(torch.tensor(x_val), torch.tensor(y_val))
        j += 1
    i += 1
    j = 0

ax = plt.subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
