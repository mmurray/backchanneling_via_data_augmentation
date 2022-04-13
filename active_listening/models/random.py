import math
import numpy as np
from scipy.stats.distributions import expon, norm


# Start/duration statistics from the dataset
DURATION_EXP_LOC = 2.348721902017291
DURATION_EXP_SCALE = 1.2706147139049553
START_EXP_LOC = 1.4000000000000004 
START_EXP_SCALE = 7.531469740634007


def get_random_start_time():
    return expon.rvs(START_EXP_LOC, START_EXP_SCALE)


def get_random_duration():
    return norm.rvs(DURATION_EXP_LOC, DURATION_EXP_SCALE)
