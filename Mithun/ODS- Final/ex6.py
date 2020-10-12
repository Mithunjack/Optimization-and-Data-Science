import numpy as np
from numpy import linalg as LA
import scipy
import itertools as it

def compute_gradient(gradient_f,values):
    grad = []
    for derivative in gradient_f:
        gradient = derivative(*values)
        grad.append(gradient)
    return np.array(grad)

def compute_hessian(hessian_matrix,values):
    hessian = []
    for row in hessian_matrix:
        row_val = []
        for element in row:
            row_value = element(*values)
            row_val.append(row_value)
        hessian.append(row_val)
    return np.array(hessian)


def descent_direction(gradient_f,values):
    grad = compute_gradient(gradient_f,values)
    descent_direction =  -(LA.norm(grad))**2
    return descent_direction


def armijo_step_algorithm(function, gradient, negative_grad, values, delta):
    step_size = 1
    descent = descent_direction(gradient, values)
    while function(*(values + step_size * negative_grad)) <= function(*values) + step_size * delta * descent:
        step_size *= 2

    while function(*(values + step_size * negative_grad)) > function(*values) + step_size * delta * descent:
        step_size /= 2
    return step_size


def set_gradient_direction(function, gradient, hessian, values):
    dk = []
    grad = compute_gradient(gradient, values)
    hess = compute_hessian(hessian, values)
    if check_gradient_relatedness(hess):
        dk = - np.dot(grad, LA.inv(hess))
        print('used newton')
    else:
        dk = - grad
        print('used gradient')

    return dk


def check_gradient_relatedness(matrix):
    eigen_values = LA.eigvalsh(matrix)
    if eigen_values[0]/eigen_values[len(eigen_values)-1] > 0:
        return True
    else:
        return False


def general_descent(function,gradient_f,initial,use_newton_direction,stopping_creiterion,max_iteration):
    xk = initial
    counter = 0
    xk1 = 0
    delta = 10**(-4)
    if use_newton_direction:
        xk1 = globalized_newton_method(function,gradient,hessian,initial,stopping_creiterion,max_iteration)
    else:
        while LA.norm(compute_gradient(gradient,xk)) > 0.01:
            dk = -compute_gradient(gradient,xk)
            step_size = armijo_step_algorithm(function,gradient,dk,xk,delta)
            xk1 = xk + step_size*dk
            xk = xk1
            print('xk1 at iterate: {}--->: {}'.format(counter,xk1))
            print('dk at iterate: {}--->: {}'.format(counter,dk))
            print('step_size at iterate: {}--->: {}'.format(counter,step_size))
            counter += 1
            if counter == max_iteration:
                break
    return xk1

def globalized_newton_method(function,gradient,hessian,initial,stopping_creiterion,max_iteration):
    xk = initial
    counter = 0
    xk1 = 0
    delta = 10**(-4)
    while LA.norm(compute_gradient(gradient,xk)) > 0.01:
        dk = set_gradient_direction(function,gradient,hessian,xk)
        step_size = armijo_step_algorithm(function,gradient,dk,xk,delta)
        xk1 = xk + step_size*dk
        xk = xk1
        print('xk1 at iterate: {}--->: {}'.format(counter,xk1))
        print('dk at iterate: {}--->: {}'.format(counter,dk))
        print('step_size at iterate: {}--->: {}'.format(counter,step_size))
        counter += 1
        if counter == max_iteration:
                break
    return xk1

if __name__ == "__main__":
    function = lambda x, y: 100 * (y - x * x) ** 2 + (1 - x) ** 2
    dfx = lambda x, y: -400 * x * (y - x * x) + 2 * x - 2
    dfy = lambda x, y: 200 * (y - x * x)

    dxdx = lambda x, y: (-400 * (y - 3 * x ** 2)) + 2
    dxdy = lambda x, y: -400 * x
    dydx = lambda x, y: -400 * x
    dydy = lambda x, y: 200

    gradient = np.array([dfx, dfy])
    hessian = np.array([np.array([dxdx, dydx]), np.array([dxdy, dydy])])

    minimizer = general_descent(function, gradient, np.array([3, 2]), False, 0.01, 20000)
    print('=================MINIMIZER===============', minimizer)
    minimum = function(*minimizer)
    print('=================MINIMMUM===============', minimum)

    # x_star = globalized_newton_method(function, gradient, hessian, np.array([3, 2]), True, 7000)
    # print(x_star)