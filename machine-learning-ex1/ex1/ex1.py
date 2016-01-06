import numpy as np
import scipy
from matplotlib import pyplot, cm
import copy

def warm_up_exercises():
    return np.identity(5)

def part_2():
    data = load_text_data('ex1data1.txt')
    x,y = data[:,0], data[:,1]
    m = len(y)
    #plot(x,y)

def part_3_gradient_descent():
    data = load_text_data('ex1data1.txt')
    X,y = data[:,0], data[:,1]
    m = len(y)
    X = np.hstack([np.ones((m,1)), X.reshape(m,1)])

    theta = np.zeros((2,1))
    iterations = 1500
    alpha = 0.01

    cost = compute_cost(X,y,theta)
    theta = gradient_descent(X,y,theta,alpha,iterations)

    print np.array([1,3.5]).dot(theta)
    print np.array([1,7]).dot(theta)

    #plot(X[:,1],y)
    #pyplot.plot(X[:,1], X.dot(theta), 'b-')
    #pyplot.show()

def compute_cost(x,y,theta):
    m = len(y)
    sumed = sum(map(lambda a: (hypothesis(x[a], theta) - y[a])**2, range(m)))
    return 1/(2*float(m)) * sumed

#calculates guess of output based on currently calculated thetas
def hypothesis(x, theta):
    return x.dot(theta)

#this continually runs iterations on the dataset constantly updating the
#thetas(coefficients) to better fit
def gradient_descent(X,y,theta, alpha, iterations):
    m = len(y)
    j_history = []

    #changes beta by the average difference between hypothesis and actual
    #expected output. then muliplies by alpha to scale it down
    for it in xrange(iterations):
        theta[0] -= alpha/float(m)*sum(map(lambda a: (hypothesis(X[a], theta)
                                                  - y[a]), range(m)))
        theta[1] -= alpha/float(m)*sum(map(lambda a: (hypothesis(X[a], theta)
                                                  - y[a])*X[a,1], range(m)))
        print theta
    return theta

def load_text_data(f):
    return np.genfromtxt(f, delimiter=',')

def plot(x, y):
    pyplot.plot(x,y, 'rx', markersize=5)
    pyplot.ylabel('Profit in $10,000s')
    pyplot.xlabel('Population of City in 10,000s')
    pyplot.show()




if __name__ == '__main__':
    warm_up_exercises()
    part_2()
    part_3_gradient_descent()


