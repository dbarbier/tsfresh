import os
from multiprocessing import cpu_count

n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())

CHUNKSIZE = None
N_PROCESSES = max(1, n_cores // 2)
PROFILING = False
PROFILING_SORTING = "cumulative"
PROFILING_FILENAME = "profile.txt"
IMPUTE_FUNCTION = None
CUSTOM_FUNCTIONS = None
DISABLE_PROGRESSBAR = False
SHOW_WARNINGS = False
PARALLELISATION = True
TEST_FOR_BINARY_TARGET_BINARY_FEATURE = "fisher"
TEST_FOR_BINARY_TARGET_REAL_FEATURE = "mann"
TEST_FOR_REAL_TARGET_BINARY_FEATURE = "mann"
TEST_FOR_REAL_TARGET_REAL_FEATURE = "kendall"
FDR_LEVEL = 0.05
HYPOTHESES_INDEPENDENT = False
WRITE_SELECTION_REPORT = False
RESULT_DIR = "logging"
