import numpy as np


def part_one():
    data = load_text_data('ex2data1.txt')
    m, n = np.shape(data)
    n -= 1
    X, y = data[:,:n], data[:,n:]
    print X
    print y




def load_text_data(f):
    return np.genfromtxt(f, delimiter=',')

if __name__ == '__main__':
    part_one()
