import sys
import contextlib
import subprocess


@contextlib.contextmanager
def profiling(filename=None, stream=sys.stdout, sort_stats='cumulative', snakeviz=True):
    """Context-manager to do profiling on a very specific piece of code.

    Parameters
    ----------
    filename : str, optional
        filename to save output to
    stream : optional
        steam to stream output to
    sort_stats : str, optional
        See  pstats.Stats documentation:
            The sort_stats() method now processes some additional options (i.e., in
            addition to the old -1, 0, 1, or 2).  It takes an arbitrary number of
            quoted strings to select the sort order.  For example sort_stats('time',
            'name') sorts on the major key of 'internal function time', and on the
            minor key of 'the name of the function'.  Look at the two tables in
            sort_stats() and get_sort_arg_defs(self) for more examples.
    snakeviz : bool, optional
        if True, a snakeviz visualisation of the output is shown

    Notes
    -----
    The output file can be visualized with tools such as f.i. snakeviz

    """
    import cProfile     # noqa; only need this in debug scripts, not in production
    import pstats       # noqa

    pr = cProfile.Profile()
    # start profiling
    pr.enable()

    yield

    # stop profiling
    pr.disable()
    # collect the stats
    stats = pstats.Stats(pr, stream=stream).sort_stats(sort_stats)
    if stream:
        # print to the stream
        stats.print_stats()
    if snakeviz and not filename:
        filename = 'temp.prof'
    if filename:
        # dump to file
        stats.dump_stats(filename)
    if snakeviz:
        try:
            subprocess.Popen('snakeviz {}'.format(filename)).wait(1)
        except subprocess.TimeoutExpired:
            # NOTE : Give the snakeviz server some time to start up
            pass
