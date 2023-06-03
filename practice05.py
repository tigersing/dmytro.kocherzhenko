import torch
import torch.optim as optim

def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

def compute_gradients(x, y):
    x.requires_grad_()
    y.requires_grad_()
    z = matyas(x, y)
    gradients = torch.autograd.grad(z, (x, y), create_graph=True)
    return gradients

def gradient_descent_momentum(lr, momentum, num_iters):
    x = torch.tensor(0.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    velocity_x = torch.tensor(0.0)
    velocity_y = torch.tensor(0.0)

    for i in range(num_iters):
        gradients = compute_gradients(x, y)
        gradient_x, gradient_y = gradients

        velocity_x = momentum * velocity_x - lr * gradient_x
        velocity_y = momentum * velocity_y - lr * gradient_y

        x.data += velocity_x
        y.data += velocity_y

    return x.item(), y.item()

def optimize_with_pytorch(lr, momentum, num_iters):
    x = torch.tensor(0.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    optimizer = optim.SGD([x, y], lr=lr, momentum=momentum, nesterov=False)

    for i in range(num_iters):
        optimizer.zero_grad()
        z = matyas(x, y)
        z.backward()
        optimizer.step()

    return x.item(), y.item()

LR = 0.1
DEFAULT_MOMENTUM = 0.9
NUM_ITERS = 100

custom_x, custom_y = gradient_descent_momentum(LR, DEFAULT_MOMENTUM, NUM_ITERS)
torch_x, torch_y = optimize_with_pytorch(LR, DEFAULT_MOMENTUM, NUM_ITERS)

print("Custom implementation - Final values:")
print(f"x: {custom_x}, y: {
