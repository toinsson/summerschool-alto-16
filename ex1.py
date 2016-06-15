
keypad = [
(0 ,0),
(-1,3),
(0 ,3),
(1 ,3),
(-1,2),
(0 ,2),
(1 ,2),
(-1,1),
(0 ,1),
(1 ,1)
]

from scipy.stats import norm

gaussians = [(norm(x), norm(y)) for (x,y) in keypad]

def single_likelihood(key, touch):
    nx, ny = gaussians[key]
    return nx.pdf(touch[0]) * ny.pdf(touch[1])


def likelihoods(touch):
    res = []
    for key in range(10):  ## all keys on keypad
        likelihood = single_likelihood(key, touch)
        res.append((str(key), likelihood))
    return res


def get_weighted_codes(touches):
    start_prior = [('e', 1)]
    res = recursive(start_prior, touches)
    return res

import itertools

def recursive(prior, touches):

    t0 = touches[0]
    l0 = likelihoods(t0)

    res = []
    for product in itertools.product(*[prior, l0]):
        (k0, p0), (k1, p1) = product
        res.append((k0+k1, p0*p1))

    if len(touches) == 1:
        return res
    else:
        return recursive(res, touches[1:])

import numpy as np

def sorted_weighted_codes(touches):
    res = get_weighted_codes(touches)
    res_d = dict()
    for x,y in enumerate(res):
        res_d[y[0]] = y[1]

    import operator
    sorted_x = sorted(res_d.items(), key=operator.itemgetter(1))

    return sorted_x


def main():
    # touches = [[0,0],[1,1]]
    # res = get_weighted_codes(touches)

    # res_d = dict()
    # for x,y in enumerate(res):
    #     res_d[y[0]] = y[1]

    # import operator
    # sorted_x = sorted(res_d.items(), key=operator.itemgetter(1))

    # print sorted_x
    keys = [3,7,0,6]

    touches = [keypad[x] for x in keys]
    print sorted_weighted_codes(touches)[-5:]

    ntouches = 4

    touchx = np.random.uniform(-2, 2, size = ntouches)
    touchy = np.random.uniform(-1, 4, size = ntouches)

    touches = np.vstack((touchx, touchy)).T

    print sorted_weighted_codes(touches)[:10]

if __name__ == '__main__':
    main()

# ('e3706', 1.406107553407254e-11)