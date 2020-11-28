# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:26:39 2020

@author: Sashka
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
import numba
import time

from PDE_solver import solution_interp
from PDE_solver import string_reshape
from PDE_solver import apply_const_operator
from PDE_solver import plot_3D_surface
from PDE_solver import operator_norm


def callback_3D(Xi):
    X, T = np.meshgrid(t, x)
    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    surf = Xi.reshape([len(grid[0]), len(grid[1])])
    error = np.abs(wolfram_interp - surf)
    wolfram_MAE = np.mean(error)
    plt.title('Wolfram MAE= ' + '{:.9f}'.format(wolfram_MAE))
    surfplot = ax.plot_surface(X, T, surf, vmin=-1, vmax=1)


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='coolwarm')
    plt.colorbar()
    plt.show()


def callback_wolfram_comp(Xi, convergence=1):
    surf = Xi.reshape([len(grid[0]), len(grid[1])])
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(surf, cmap='coolwarm')
    plt.colorbar()
    f.add_subplot(1, 2, 2)
    plt.imshow(wolfram_interp, cmap='coolwarm')
    plt.colorbar()
    plt.show(block=True)
    return False


def callback_wolfram_error(Xi, convergence=1):
    surf = Xi.reshape([len(grid[0]), len(grid[1])])
    error = np.abs(wolfram_interp - surf)
    max_err_val = np.max(error)
    mean_err_val = np.mean(error)
    max_err_list.append(max_err_val)
    # mean_err_list.append(mean_err_val)
    print('max err= ',max_err_val,' mean err= ',mean_err_val)
    plt.imshow(error, cmap='coolwarm',vmin=0,vmax=max_err)
    plt.title("Error")
    plt.colorbar()
    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.plot(max_err_list)
    # f.add_subplot(1, 2, 2)
    # plt.plot(mean_err_list, c='orange')
    plt.show()
    return False


def callback_wolfram_error_anneal(Xi, f, stat):
    surf = Xi.reshape([len(grid[0]), len(grid[1])])
    error = np.abs(wolfram_interp - surf)
    max_err_val = np.max(error)
    mean_err_val = np.mean(error)
    max_err_list.append(max_err_val)
    mean_err_list.append(mean_err_val)
    # print('max err= ',max_err_val,' mean err= ',mean_err_val)
    # plt.imshow(error, cmap='coolwarm',vmin=0,vmax=max_err)
    # plt.colorbar()
    if len(max_err_list) < 50:
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.plot(max_err_list)
        f.add_subplot(1, 2, 2)
        plt.plot(mean_err_list, c='orange')
        plt.show()
    else:
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.plot(max_err_list[-50:])
        f.add_subplot(1, 2, 2)
        plt.plot(mean_err_list[-50:], c='orange')
        plt.show()
    return False


plt.rcParams["figure.max_open_warning"] = 1000

arr = []
string = []

x = np.linspace(0, 1, 21)
t = np.linspace(0, 1, 21)

grid = numba.typed.List()

grid.append(x)
grid.append(t)

wolfram = np.genfromtxt('wolfram.csv', delimiter=',')

wolfram_grid = [np.linspace(0, 1, 1001), np.linspace(0, 1, 1001)]

wolfram_interp = solution_interp(wolfram_grid, wolfram, grid)
arr = np.random.random((len(grid[0]), len(grid[1])))

# arr=np.load('sln31up4.npy')

max_err = np.max(np.abs(wolfram_interp - arr))

max_err_list = []
mean_err_list = []

part_sln = np.zeros_like(arr)

# part_sln[10:,0:11]=sln2
# part_sln[0:11,10:]=sln3
# part_sln[10:,10:]=sln4

# plot_3D_surface(part_sln,None,grid)

wolfram_interp = solution_interp(wolfram_grid, wolfram, grid)

# plot_3D_surface(wolfram_interp, None, grid)

bcond = [{'boundary': 0, 'axis': 0, 'string': np.zeros(len(grid[0]))},
         {'boundary': -1, 'axis': 0, 'string': np.zeros(len(grid[0]))},
         {'boundary': 0, 'axis': 1, 'string': np.sin(np.pi * grid[0])},
         {'boundary': -1, 'axis': 1, 'string': np.sin(np.pi * grid[0])}]

opt = minimize(operator_norm, arr.reshape(-1), args=(grid, [[(1, 0, 2, 1)], [(-1 / 4, 1, 2, 1)]], 1, bcond),
               options={'disp': True, 'maxiter': 1000}, tol=0.05)

sln = string_reshape(opt.x, grid)

full_sln_interp = solution_interp(grid, sln, grid)

plot_3D_surface(full_sln_interp, wolfram_interp, grid)

# callback_wolfram_error(opt.x)

# x_new=np.linspace(0,1,41)
# t_new=np.linspace(0,1,41)
# 
# grid_new=[x_new,t_new]

# sln_interp=solution_interp(grid,sln,grid_new)

# X, T = np.meshgrid(x_new, t_new)
# fig1 = plt.figure()
# ax = fig1.gca(projection='3d')
# surf = ax.plot_surface(X, T, sln_interp, rstride=1, cstride=1,
#     linewidth=0.1, antialiased=False)

# np.save('sln41up3',sln_interp)


# wolfram = np.genfromtxt('wolfram.csv', delimiter=',')


# X, T = np.meshgrid(x, t)
# fig1 = plt.figure()
# ax = fig1.gca(projection='3d')
# surf = ax.plot_surface(X, T, wolfram, rstride=1, cstride=1,
#     linewidth=0, antialiased=False,cmap=plt.cm.coolwarm)

# X, T = np.meshgrid(x, t)
# fig1 = plt.figure()
# ax = fig1.gca(projection='3d')
# surf = ax.plot_surface(X, T, wolfram-sln, rstride=1, cstride=1,
#     linewidth=0, antialiased=False,cmap=plt.cm.coolwarm)

# print('Error= ',np.linalg.norm(wolfram-sln))
