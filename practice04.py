import torch
import torch.optim as optim


def matyas_function (x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


# Define the starting point for optimization
x = torch.tensor([1.0, 1.0], requires_grad=True)

# Define the optimizer
optimizer = optim.SGD([x], lr=0.1)

# Define convergence criterion
threshold = 1e-6

# Optimization loop
iteration = 0
while True:
    optimizer.zero_grad()  # Reset gradients

    # Compute the value of the Matyas function and backward pass
    loss = matyas_function(x)
    loss.backward()

    optimizer.step()  # Update parameters using the gradients

    # Print the progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration: {iteration}, Loss: {loss.item()}")

    # Check convergence criterion
    if loss.item() < threshold:
        break

    iteration += 1

# Print the optimized solution
print("\nOptimized solution:")
print("x =", x.detach().numpy())
print("Matyas function value =", matyas_function(x).item())
