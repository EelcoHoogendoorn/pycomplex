
import pytest


def pytest_addoption(parser):
    parser.addoption("--show_plot", action="store", default="True",
        help="show plots: True or False")


@pytest.fixture
def show_plot(request):
    """This callable fixture allows us to either show plots when running test for visual inspection,
    or close them and at least test the absence of exceptions when running test automated"""
    try:
        flag = request.config.getoption("--show_plot")
    except:
        import os
        flag = os.environ['SHOW_PLOT']
    import matplotlib.pyplot as plt
    if flag == 'True':
        return plt.show
    else:
        return lambda: plt.close('all')
