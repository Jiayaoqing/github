# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:30:32 2018

@author: JYQ
"""

# -*- coding: utf-8 -*-
import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as amat

"this function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2"


def Rosenbrock(x, y):
    #return np.power(1 - x, 2) + np.power(100 * (y - np.power(x, 2)), 2)
    #return x*x*x - y*y*y + 3*x*x + 3*y*y + -9*x
    return x ** 2 



def show(X, Y, func=Rosenbrock):
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(X, Y, sparse=True)
    Z = func(X, Y)
    plt.title("gradeAscent image")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', )
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    plt.show()


def drawPaht(px, py, pz, X, Y, func=Rosenbrock):
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(X, Y, sparse=True)
    Z = func(X, Y)
    plt.title("gradeAscent image")
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', )
    ax.plot(px, py, pz, 'r.')  # 绘点
    plt.show()


def gradeAscent(X, Y, Maxcycles=1000, learnRate=0.0001):
    # x, Y = np.meshgrid(X, Y, sparse=True)
    new_x = [X]
    new_Y = [Y]
    g_z=[Rosenbrock(X, Y)]
    current_x = X
    current_Y = Y
    for cycle in range(Maxcycles):
        "为了更好的表示grad,我这里对表达式不进行化解"
        #current_Y -= learnRate * 200 * (Y - X * X)  #对y求导
        #current_x -= learnRate * (-2 * (1 - X) - 400 * X * (Y - X * X))  #对x求导
        current_Y -= 0  #对y求导
        current_x -= learnRate * (2 * x)  #对x求导
        X = current_x
        Y = current_Y
        new_x.append(X)
        new_Y.append(Y)
        g_z.append(Rosenbrock(X, Y))
    return new_x, new_Y, g_z


if __name__ == '__main__':
    X = np.arange(-3, 4, 0.1)
    Y = np.arange(-3, 4, 0.1)
    x = random.uniform(-3, 4)
    y = random.uniform(-3, 4)
    #print (x,y)
    x, y, z = gradeAscent(x, y)
    #print (len(x),x)
    #print (len(y),y)
    #print (len(z),z)
    drawPaht(x, y, z, X, Y, Rosenbrock)
    print(x[69], Y[69])
