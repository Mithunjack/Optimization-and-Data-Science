from numpy import linalg as LA
import numpy as np

def gradient(gradient_f,values):
    grad = []
    for eachFunction in gradient_f:
        gradient = eachFunction(*values)
        print(gradient)
        grad.append(gradient)
    return np.array(grad)

def descent_direction(gradient_f,values):
    grad = gradient(gradient_f,values)
    p = -(LA.norm(grad))**2
    return p


def armijo_step_size(f, gradient, negative_grad, values, delta):
    ro = 1
    descent = descent_direction(gradient, values)
    while f(*(values + ro * negative_grad)) <= f(*values) + ro * delta * descent:
        ro *= 2

    while f(*(values + ro * negative_grad)) > f(*values) + ro * delta * descent:
        ro /= 2
    return ro

def gradient_method(given_function, gradient_f, initial_coordinate):
    xk = initial_coordinate
    counter = 0
    xk1 = 0
    delta = 10**(-4)
    while counter <=5000:
        dk = -gradient(gradient_f,xk)
        step_size = armijo_step_size(given_function,gradient_f,dk,xk,delta)
        xk1 = xk + step_size*dk
        xk = xk1
        print('xk1 at iterate: {}--->: {}'.format(counter,xk1))
        print('dk at iterate: {}--->: {}'.format(counter,dk))
        print('step_size at iterate: {}--->: {}'.format(counter,step_size))
        counter += 1
        if LA.norm(gradient(gradient_f,xk)) <= 0.001:
            break
    return xk1

if __name__ == "__main__":
    # Roosenbork Funtion
    f = lambda x, y: 100 * (y - x * x) ** 2 + (1 - x) ** 2
    dfx = lambda x, y: -400 * x * (y - x * x) + 2 * x - 2
    dfy = lambda x, y: 200 * (y - x * x)

    # f = lambda x,y : 4*x*x+y*y
    # dfx = lambda x,y: 8*x+y*0
    # dfy = lambda x,y: 2*y+x*0
    negative_grad = gradient(np.array([dfx, dfy]), np.array([1, 2]))
    delta = 10e-4
    step_size = armijo_step_size(f, np.array([dfx, dfy]), negative_grad, np.array([1, 2]), delta)
    print(step_size)

    # excercise 2/3
    minimum_value = gradient_method(f,np.array([dfx,dfy]),np.array([1,2]))
    print('Starting point: ', np.array([1,2]))
    print('Minimum Value: ',minimum_value)