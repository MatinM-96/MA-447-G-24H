import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand, seed

# Define five different objective functions and their derivatives
def objective1(x):
    return x**3 + (6*x)**2 - (3*x) - 5

def derivative1(x):
    return 3*x**2 + 12*x - 3

def objective2(x):
    return x**3 + (5*x)**2 - (4*x) + 2

def derivative2(x):
    return 3*x**2 + 10*x - 4



# Mapping of choices to functions
objectives = [objective1, objective2]
derivatives = [derivative1, derivative2]

# Gradient descent functions
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    solutions, scores = [], []
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    for i in range(n_iter):
        gradient = derivative(solution)
        solution = solution - step_size * gradient
        solution_eval = objective(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('>%d f(%.5f) = %.5f' % (i, solution[0], solution_eval))
    return solutions, scores

def gradient_descent_with_momentum(objective, derivative, bounds, n_iter, step_size, momentum):
    solutions, scores = [], []
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    change = 0.0
    for i in range(n_iter):
        gradient = derivative(solution)
        new_change = momentum * change - step_size * gradient
        solution = solution + new_change
        change = new_change
        solution_eval = objective(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('>%d f(%.5f) = %.5f' % (i, solution[0], solution_eval))
    return solutions, scores

def plot_results(objective, solutions, scores, bounds):
    inputs = np.arange(bounds[0, 0], bounds[0, 1] + 0.1, 0.1)
    results = objective(inputs)
    plt.plot(inputs, results)
    plt.plot(solutions, scores, '.-', color='red')
    plt.show()
def runOppgave1_3():
    seed(4)
    bounds = np.asarray([[-1.0, 1.0]])
    n_iter = 30
    step_size = 0.1
    momentum = 0.3

    while True:
        # Prompt user to select an equation
        print("\nChoose an equation for:")
        for i in range(1, 3):
            print(f"{i}: Objective {i}")
        print("0: Exit")
        choice = int(input("Enter the number of the equation (1-2) or 0 to exit: "))

        if choice == 0:
            print("Exiting program.")
            break

        elif 1 <= choice <= 2:
            selected_objective = objectives[choice - 1]
            selected_derivative = derivatives[choice - 1]

            # Perform gradient descent without momentum
            print("\nGradient Descent without Momentum:")
            solutions, scores = gradient_descent(selected_objective, selected_derivative, bounds, n_iter, step_size)
            plot_results(selected_objective, solutions, scores, bounds)

            # Perform gradient descent with momentum
            print("\nGradient Descent with Momentum:")
            solutions, scores = gradient_descent_with_momentum(selected_objective, selected_derivative, bounds, n_iter, step_size, momentum)
            plot_results(selected_objective, solutions, scores, bounds)

        else:
            print("Invalid choice. Please select a number between 1 and 2 or 0 to exit.")