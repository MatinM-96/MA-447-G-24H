from numpy import arange
from matplotlib import pyplot
from numpy import asarray
from numpy.random import rand
from numpy.random import seed


def objective(x):
    return x**3+(6*x)**2-(3*x)-5
def derivative(x):
    return 3*x**2+12*x-3

def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution)
        # take a step
        solution = solution - step_size * gradient
        # evaluate candidate point
        solution_eval = objective(solution)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
        # report progress
        print('>%d f(%.5f) = %.5f' % (i, solution[0], solution_eval))
    return [solutions, scores]

# Define gradient descent with momentum function
def gradient_descent_with_momentum(objective, derivative, bounds, n_iter, step_size, momentum):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # keep track of the change
    change = 0.0
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution)
        # calculate update
        new_change = momentum * change - step_size * gradient
        # take a step
        solution = solution + new_change
        # save the change
        change = new_change
        # evaluate candidate point
        solution_eval = objective(solution)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
        # report progress
        print('>%d f(%.5f) = %.5f' % (i, solution[0], solution_eval))
    return [solutions, scores]




def runOppgave1_2():
    # Seed the pseudo random number generator
    seed(4)
    # Define range for input
    bounds = asarray([[-1.0, 1.0]])
    # Define the total iterations
    n_iter = 30
    # Define the step size
    step_size = 0.1
    # Perform the gradient descent search
    solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
    # Sample input range uniformly at 0.1 increments
    inputs = arange(bounds[0, 0], bounds[0, 1] + 0.1, 0.1)
    # Compute targets
    results = objective(inputs)
    # Create a line plot of input vs result
    pyplot.plot(inputs, results)
    # Plot the solutions found
    pyplot.plot(solutions, scores, '.-', color='red')
    # Show the plot
    pyplot.show()



    # Seed the pseudo random number generator
    seed(4)
    # Define the total iterations
    n_iter = 30
    # Define the step size
    step_size = 0.1
    # Define momentum
    momentum = 0.3
    # Perform the gradient descent search with momentum
    solutions, scores = gradient_descent_with_momentum(objective, derivative, bounds, n_iter, step_size, momentum)
    # Sample input range uniformly at 0.1 increments
    inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
    # Compute targets
    results = objective(inputs)
    # Create a line plot of input vs result
    pyplot.plot(inputs, results)
    # Plot the solutions found
    pyplot.plot(solutions, scores, '.-', color='red')
    # Show the plot
    pyplot.show()