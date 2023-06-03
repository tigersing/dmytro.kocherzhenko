import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Matyas_function(nn.Module):
    def __init__(self, x0, y0, device="cpu"):
        super(Matyas_function, self).__init__()
        self.x = nn.Parameter(x0)
        self.y = nn.Parameter(y0)

    def forward(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        loss = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return loss

    def string(self):
         return f'x = {self.x.item()} y = {self.y.item()}'

model = Matyas_function(torch.tensor(4.0), torch.tensor(4.0))

x = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, x)

Z = np.empty((50, 50), dtype=np.float32)
i, j = 0, 0
for x in np.linspace(-10, 10, 50):
    for y in np.linspace(-10, 10, 50):
        Z[i, j] = model.forward(torch.tensor(x), torch.tensor(y))
        j += 1
    i += 1
    j = 0

ax = plt.subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
