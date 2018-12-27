import numpy as np

if __name__ == '__main__':
    a = [1, 2, 2, 3, 4]
    d = [3, 3, 4]
    rand_idx = np.random.permutation(len(a))
    print(rand_idx)