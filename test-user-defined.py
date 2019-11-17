import tsfresh
from tsfresh.feature_extraction.feature_calculators import set_property, value_count
import pandas as pd
import numpy as np
from collections import Counter

np.random.seed(42)
df = pd.DataFrame(np.concatenate([np.repeat(range(200),10)[:,np.newaxis],
                                  np.tile(range(10),200)[:,np.newaxis],
                                  np.random.randn(2000, 1)], axis=1), columns=["id", "time", "x"])

# Custom functions to add
@set_property("fctype", "simple")
def value_count_all(x):
    """
    Returns the number of values in x

    :param x: the time series on which to calculate the feature.
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list
    """
    values, counts = np.unique(x, return_counts=True)

    return [("value_count__value_\"{}\"".format(value), value_count(x, value))
            for value in values]


@set_property("fctype", "simple")
def last(x):
    """Return the last value of x.

    :param x: the time series on which to calculate the feature.
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list
    """
    return x[-1]


@set_property("fctype", "simple")
def first(x):
    """Return the first value of x.

    :param x: the time series on which to calculate the feature.
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list
    """
    return x[0]


@set_property("fctype", "simple")
def is_measured(x):
    """
    Check if a variable has been measured – i.e. if the series is not empty.

    :param x: the time series to calculate the feature of
    :type x: np.ndarray
    :return: the different feature values
    :return type: float
    """
    return float(bool(len(x)))


@set_property("fctype", "simple")
def mode(x):
    """Return the mode of the parameter (i.e. most common value)

    :param x: the time series to calculate the feature of
    :type x: np.ndarray
    :return: the different feature values
    :return type: tuple
    """
    c = Counter(x)
    return tuple(x for x, count in c.items() if count == c.most_common(1)[0][1])


@set_property("fctype", "simple")
@set_property("minimal", True)
def count(x):
    """
    Returns the number of elements in x

    :param x: the time series to calculate the feature of
    :type x: np.ndarray
    :return: the value of this feature
    :return type: int
    """
    return len(x)

# Dictionary of custom functions; drop value_count_all and mode which returns multiple values
custom_functions = {f.__name__: f for f in [last, first, is_measured, count]}

params = tsfresh.feature_extraction.EfficientFCParameters()
params.update({fname: None for fname in custom_functions.keys()})

ts = tsfresh.extract_features(df, column_id="id", default_fc_parameters=params, custom_functions=custom_functions)
for fname in custom_functions.keys():
    print(ts['x__'+fname])
