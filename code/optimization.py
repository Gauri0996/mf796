import numpy as numpy
import pandas as pd
from scipy.optimize import minimize
from shrinkage_matrix_form import Covariance_Shrinkage


class Optimizer:

    def __init__(self, returns, benchmark = None):
        self.returns = returns