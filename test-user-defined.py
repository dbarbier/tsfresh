import tsfresh
from tsfresh.feature_extraction.feature_calculators import set_property
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame(np.concatenate([np.repeat(range(200),10)[:,np.newaxis],
                                  np.tile(range(10),200)[:,np.newaxis],
                                  np.random.randn(2000, 1)], axis=1), columns=["id", "time", "x"])

# Custom function to add
@set_property("fctype", "simple")
def count(x, a):
    return len(x) + a

# Dictionary of custom functions
custom_functions = {
     'count': count
}

params = tsfresh.feature_extraction.EfficientFCParameters()
params.update({'count': [{'a': a} for a in np.arange(5)]})

ts = tsfresh.extract_features(df, column_id="id", default_fc_parameters=params, custom_functions=custom_functions)
