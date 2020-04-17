import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shrinkage_matrix_form import Covariance_Shrinkage


class Optimizer:

    def __init__(self, returns, benchmark = None):
        self.returns = returns
        self.mean_returns = np.array([self.returns.mean()]).transpose()
        self.n = len(self.mean_returns)
        self.cons= [{'type':'eq','fun': lambda x: np.sum(x)-1}]
        self.no_short_bounds = self.n*[(0,1)]
    
    def calc_expected_portfolio_return(self,w):
        w = np.array([w]).transpose() #making w a column vector
        exp_ret = self.mean_returns.transpose() @ w 
        return exp_ret[0][0]
    
    def calc_portfolio_variance(self,w,cov_mat):
        w = np.array([w]).transpose() #making w a column vector
        variance = w.transpose() @ cov_mat @ w
        return variance[0][0]
    
    def minimize_variance(self,cov_mat, short_sales = 'no'):
        guess = self.n*[1/self.n]
        if short_sales =='yes':
            bound = None
        else:
            bound = self.no_short_bounds        
        optimized_weights = minimize(self.calc_portfolio_variance, guess, bounds = bound,constraints = self.cons, args = cov_mat, method= 'SLSQP',options={ 'ftol': 1e-09})['x']
        return optimized_weights

    def mean_variance_objective(self,w,cov_mat,a):
        objective = self.calc_expected_portfolio_return(w) - a*self.calc_portfolio_variance(w,cov_mat)
        return - objective
        
    def optmize_mean_variance(self,cov_mat,a,short_sales = 'no'):
        guess = self.n*[1/self.n]
        if short_sales =='yes':
            bound = None
        else:
            bound = self.no_short_bounds 
        optimized_weights = minimize(self.mean_variance_objective, guess, bounds = bound,constraints = self.cons, args = (cov_mat,a), method= 'SLSQP',options={ 'ftol': 1e-09})['x']
        return optimized_weights


if __name__=="__main__":
    """ Change the path to your path if you are gonna run this """

    george = r"C:\Users\jojis\OneDrive\Documents\GitHub\mf796\data\adjclosepx.csv"
    robbie = "/home/robbie/Documents/MSMF/Spring2020/MF796/project/data/adjclosepx.csv"
    issy = "/Users/issyanand/Desktop/adjclosepx.csv"
    gauri = "C:\Users\gauri\OneDrive\Desktop\Spring 2020\MF 796\Project\adjclosepx.csv"
    stock_prices = pd.read_csv(george, index_col="Date")
    stock_prices.dropna(inplace=True)
    stock_returns = stock_prices.pct_change(1)
    stock_returns = stock_returns.loc['02/01/2015':'31/12/2019',:]

    test = Covariance_Shrinkage(stock_returns)
    shrunk_mat = test.get_shrunk_cov_mat()
    cov = test.get_sample_cov_mat()
    opt = Optimizer(stock_returns)
    print(opt.optmize_mean_variance(shrunk_mat,2, short_sales='no'))
    print(opt.optmize_mean_variance(cov,2, short_sales='no'))
