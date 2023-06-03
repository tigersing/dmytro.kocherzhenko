import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Определение функции Матиоша
def matyas(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

# Функция для отслеживания значений функции в каждой итерации
def track_convergence(result):
    values = []
    for iteration in result['nit']:
        values.append(result['fun'])
    return values

# Метод Нелдера-Мида (Nelder-Mead)
def nelder_mead():
    return minimize(matyas, [0, 0], method='Nelder-Mead', options={'disp': True})

# Метод BFGS
def bfgs():
    return minimize(matyas, [0, 0], method='BFGS', options={'disp': True})

# Метод L-BFGS-B
def l_bfgs_b():
    return minimize(matyas, [0, 0], method='L-BFGS-B', options={'disp': True})

# Запуск и отслеживание сходимости каждого алгоритма
nelder_mead_result = nelder_mead()
bfgs_result = bfgs()
l_bfgs_b_result = l_bfgs_b()

nelder_mead_convergence = track_convergence(nelder_mead_result)
bfgs_convergence = track_convergence(bfgs_result)
l_bfgs_b_convergence = track_convergence(l_bfgs_b_result)

# Построение графиков сходимости
plt.plot(nelder_mead_convergence, label='Nelder-Mead')
plt.plot(bfgs_convergence, label='BFGS')
plt.plot(l_bfgs_b_convergence, label='L-BFGS-B')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.legend()
plt.show()


# В этой секции мы будем минимизировать с помощью стандартного SGD алгоритма с моментом и уточнением Нестерова

def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

def grad_matyas(x, y):
    grad_x = 0.52 * x - 0.48 * y
    grad_y = 0.52 * y - 0.48 * x
    return grad_x, grad_y

def sgd_momentum_nesterov(lr, momentum, nesterov, num_epochs):
    x = 0
    y = 0
    velocity_x = 0
    velocity_y = 0
    trajectory = []

    for epoch in range(num_epochs):
        grad_x, grad_y = grad_matyas(x, y)
        velocity_x = momentum * velocity_x - lr * grad_x
        velocity_y = momentum * velocity_y - lr * grad_y

        if nesterov:
            x_tilde = x + momentum * velocity_x
            y_tilde = y + momentum * velocity_y
            grad_x_tilde, grad_y_tilde = grad_matyas(x_tilde, y_tilde)
            velocity_x = momentum * velocity_x - lr * grad_x_tilde
            velocity_y = momentum * velocity_y - lr * grad_y_tilde

        x += velocity_x
        y += velocity_y
        trajectory.append(matyas(x, y))

    return trajectory

lr = 0.1
momentum = 0.9
nesterov = True
num_epochs = 100

trajectory = sgd_momentum_nesterov(lr, momentum, nesterov, num_epochs)

plt.plot(trajectory)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('SGD with Momentum and Nesterov')
plt.show()

# В этой секции мы будем минимизировать с помощью стандартного алгоритма Adam
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def matyas(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

# Задаем начальные значения
x0 = [0, 0]

# Минимизируем функцию с помощью алгоритма Adam
result = minimize(matyas, x0, method='Nelder-Mead', options={'disp': True})

# Получаем оптимальные значения
x_opt = result.x

# Выводим оптимальные значения
print("Оптимальные значения:")
print(f"x: {x_opt[0]}, y: {x_opt[1]}")

# Построение графика функции Matyos
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = matyas([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.title('Matyas Function')
plt.show()



