# author: Joscha Reimer
# This is an example code snippet that shows how to perform a minimization and especially how to plot the results.
# (In particular, the module general_descent_method must be implemented appropriately.)

import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D # needed for projection='3d'
import matplotlib.pyplot as plt

import general_descent_method


def run_minimization(function, x0, descent_method,
                     descent_method_parameters={}, step_size_method_parameters={}):
    function_name = function.NAME
    f = function.f
    D_f = function.D_f
    D2_f = function.D2_f
    x_min = function.X_MIN
    f_x_min = f(x_min)
    n = len(x_min)

    # minimize
    all_x = []
    def callback(x):
        all_x.append(np.array(x))

    x, f_x, f_count, D_f_count, D2_f_count, iteration_count = general_descent_method.minimize(x0, f, D_f, D2_f,
        descent_method=descent_method, descent_method_parameters=descent_method_parameters,
        step_size_method_parameters=step_size_method_parameters, callback=callback)
    total_amount = f_count + n * D_f_count + n**2 * D2_f_count

    # prepare and print description
    description = f'{function_name} was minimized using {descent_method} starting from {x0}'
    if len(descent_method_parameters) > 0:
         description += f' with parameters {descent_method_parameters}'
    if len(step_size_method_parameters) > 0:
        if len(descent_method_parameters) > 0:
            description += ' and'
        description += f' with step-size parameters {step_size_method_parameters}'
    description += '.' + os.linesep
    description += f'Minimum found at {x} with function value {f_x}. '
    description += f'{iteration_count} iterations were needed with {f_count} function, {D_f_count} derivative and {D2_f_count} second derivative evaluations. '
    description += f'The total amount of work was thus {total_amount}.'
    print(description)
    
    # get all x and all f_x
    all_x = np.array(all_x).T
    all_f_x = f(all_x)

    # prepare data for surface plot
    offset = 0.5
    plot_step_size = 0.1
    X = np.arange(min(all_x[0].min(), x_min[0]) - offset, max(all_x[0].max(), x_min[0]) + offset, plot_step_size)
    Y = np.arange(min(all_x[1].min(), x_min[1]) - offset, max(all_x[1].max(), x_min[1]) + offset, plot_step_size)
    X_surface, Y_surface = np.meshgrid(X, Y)
    Z_surface = f([X_surface, Y_surface])

    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X_surface, Y_surface, Z_surface, cmap=mpl.cm.coolwarm, alpha=0.5)
    ax.plot([x_min[0]], [x_min[1]], [f_x_min], '*', color='g', markersize=10)
    ax.plot(all_x[0], all_x[1], all_f_x, color='k')
    ax.plot(all_x[0], all_x[1], all_f_x, '+', color='r')
    plt.title(description)

