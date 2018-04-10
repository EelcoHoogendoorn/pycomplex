import operator


def pairs(iterable):
    """Yield pairs of an iterator"""
    it = iter(iterable)
    x = next(it)
    while True:
        p = x
        x = next(it)
        yield p, x


def accumulate(iterable, func=operator.add):
    """Return running totals; backported for python 2 support"""
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total