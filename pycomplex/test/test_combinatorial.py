
from pycomplex.math.combinatorial import permutations, permutations_map, combinations

def test_parmutation_parity():

    l = [1, 2, 3, 4]
    l=[0, 1, 2]
    r = list(permutations(l))

    print(r)


def test_permutations_map():
    par, perm = permutations_map(4)
    print(par, perm)


def test_combinations():
    c = combinations([0, 1], 1)
    print(list(c))

    c = combinations([0, 1, 2], 1)
    print(list(c))

    c = combinations([0, 1, 2], 2)
    print(list(c))

    c = combinations([0, 1, 2, 3], 2)
    print(list(c))
