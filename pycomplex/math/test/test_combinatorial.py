
from pycomplex.math.combinatorial import permutations, combinations


def test_parmutation_parity():
    l = [1, 2, 3, 4]
    l = [0, 1, 2]
    r = list(permutations(l))
    print(r)


def test_combinations():
    c = combinations([0, 1], 1)
    print(list(c))

    c = combinations([0, 1, 2], 1)
    print(list(c))

    c = combinations([0, 1, 2], 2)
    print(list(c))

    c = combinations([0, 1, 2, 3], 2)
    print(list(c))

import numpy as np
import matplotlib.pyplot as plt

def get_table(N, n):
    c = list(combinations(range(N), n))

    # one hot encode
    z = np.zeros((len(c), N), dtype=np.bool)
    for i, q in enumerate(c):
        z[i, q[1]] = 1

    def f(zz, j):
        i = (z[j:j+1] * z).sum(axis=1) == 1 #keep all others with a single overlap
        assert i[:j].sum() == j     # assert we keep all previous entries
        i[j] = 1                    # keep testing entry
        return zz[i]

    def detrivialize(zz):
        # no single symbol should occur in all the cards
        n = z.sum(axis=0)

    for i in range(100):
        z = f(z, i)
        # print(i, len(z))
        if i >= len(z)-1:
            break
    return z

def test_q():
    """31 distinct animals yielding 19 cards best solution so far.
    higher numbers collapse things; what gives?
    are we incorrect to just collapse the first?
    like n=6 n=7 also caps out at 19, and breaks symmetry

    n=5 goes up to 21 cards, interestingly
    """
    a = np.array([[3, 4], [6, 8]])
    print(np.einsum('ij,jk->ik', a, a))
    quit()
    N = 7
    n = 3
    z = get_table(N, n)

    N = 3**2+3+1
    n = 4

    N = 4**2+4+1
    n = 5

    # these cap out at 19
    # N = 5**2+5+1
    # n = 6
    #
    # N = 6**2+6+1
    # n = 7

    z = get_table(N, n)

    # for N in range(12, 40):
    #     z = get_table(N, n)
    #     print(N, len(z))

        # i = (z[0:1] * z).sum(axis=1)==1
    # i[0] = 1
    # z1 = z[i]
    # i = (z1[1:2] * z1).sum(axis=1)==1
    # i[1] = 1
    # z2 = z1[i]


    # j = z1[:, None, :] * z1[None, :, :]
    # q = j.sum(axis=-1)

    # print(i.sum())
    # print(z[i])
    print(z.sum(axis=0))
    plt.imshow(z, interpolation='nearest')
    plt.show()
