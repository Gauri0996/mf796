"""
TO DO (Short term):
1. Think about efficiency when it comes to the pi and rho functions. If we could turn some of the sums into matrix multiplication this would go way faster.
   Everyone please think about this ^^^^^
2. Let's add comments and docstrings where appropriate
3. Let's switch up the file to monthly returns.
4. Start thinking about optimizing the portfolio (start a new file for that and import this class). How are we defining the optimization problem? (We need the matrices like in the HW).
   We are going to have inequality constraints. Find an optimizer that can deal with that.
5. Do 1,2,3, and 4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Covariance_Shrinkage():

    def __init__(self,cov_mat, corr_mat, ret):
        self.cov_mat = np.array(cov_mat)
        self.corr_mat = np.array(corr_mat)
        self.N = len(corr_mat[0:])
        self.ret = ret
        self.T = len(ret.iloc[:,0])
        self.mean_ret = ret.mean()
        
        r_bar_sum = 0 
        for i in range(0,self.N-1):
            for j in range(i+1,self.N):
                r_bar_sum += self.corr_mat[i,j]
        self.r_bar = (2/(self.N*(self.N-1)))*r_bar_sum

    def get_F(self):

        F = np.zeros(  (self.N,self.N)   )
        
        for i in range(0,self.N):
            for j in range(0,self.N):
                if i == j:
                    F[i,j] = self.cov_mat[i,j]
                else:
                    F[i,j] = self.r_bar * np.sqrt(self.cov_mat[i,i]*self.cov_mat[j,j])
        return F
    
    def get_gamma(self):
        gamma = 0
        F = self.get_F()
        for i in range(0,self.N):
            for j in range(0,self.N):
                gamma += ( (F[i,j] - self.cov_mat[i,j])**2 )
        return gamma

    def get_pi(self):
        pi = 0
        for i in range(0,self.N):
            for j in range(0,self.N):
                for t in range(0,self.T):
                    pi += ( (self.ret.iloc[t,i] - self.mean_ret[i])*(self.ret.iloc[t,j] - self.mean_ret[j]) - self.cov_mat[i,j] )**2
        pi  = pi * (1/self.T)
        return pi
    
    def get_rho(self):
        
        sum1 = 0
        for i in range(0,self.N):
            for j in range(0,self.N):
                if i != j:
                    theta_ii = 0
                    for t in range(0,self.T):
                        comp_1 = ((self.ret.iloc[t,i] - self.mean_ret[i])**2 - self.cov_mat[i,i]) 
                        comp_2 = ((self.ret.iloc[t,i] - self.mean_ret[i])*(self.ret.iloc[t,j] - self.mean_ret[j]) - self.cov_mat[i,j])
                        theta_ii += comp_1*comp_2
                    theta_ii *= 1/self.T
        
                    theta_jj = 0
                    for t in range(0,self.T):
                        comp_1 = ((self.ret.iloc[t,j] - self.mean_ret[j])**2 - self.cov_mat[j,j]) 
                        comp_2 = ((self.ret.iloc[t,i] - self.mean_ret[i])*(self.ret.iloc[t,j] - self.mean_ret[j]) - self.cov_mat[i,j])
                        theta_jj += comp_1*comp_2
                    theta_jj *= 1/self.T
                    
                    sum1 += (np.sqrt(self.cov_mat[j][j]/self.cov_mat[i][i])*theta_ii + np.sqrt(self.cov_mat[i][i]/self.cov_mat[j][j])*theta_jj)
        sum1 *= (self.r_bar*0.5)

        sum2 = 0
        for i in range(0,self.N):
            for t in range(0,self.T):
                sum2 += ( ((self.ret.iloc[t,i] - self.mean_ret[i])**2) -self.cov_mat[i,i] )**2
        sum2 *= 1/self.T
        
        rho = sum1 + sum2
        return rho
    
    def get_kappa(self):
        return (self.get_pi() - self.get_rho())/self.get_gamma()

    def get_delta(self):
        delta = max(0, min(self.get_kappa()/self.T,1))
        return delta

    def get_shrunk_cov_mat(self):
        delta = self.get_delta()
        shrunk_cov_mat = (delta * self.get_F()) + (1-delta)*self.cov_mat
        return shrunk_cov_mat

if __name__ == "__main__": 

    """ Change the path to your path if you are gonna run this """

    george = r"C:\Users\jojis\OneDrive\Documents\GitHub\mf796\data\adjclosepx.csv"
    robbie = "/home/robbie/Documents/MSMF/Spring2020/MF796/project/data/adjclosepx.csv"
    issy = "/Users/issyanand/Desktop/adjclosepx.csv"
    stock_prices = pd.read_csv(george, index_col="Date")
    stock_prices.dropna(inplace=True)
    stock_returns = stock_prices.pct_change(1)
    stock_returns = stock_returns.loc['01/10/2019':'31/12/2019',:]
    covariance = stock_returns.cov()
    correlation = stock_returns.corr()

    #making sure things are working
    
    #test = Covariance_Shrinkage(covariance,correlation, stock_returns)
    
    # F = test.get_F()
    # print(F)
    # gamma = test.get_gamma()
    # print(gamma)
    # pi = test.get_pi()
    # print(pi)
    # rho = test.get_rho()
    # print(rho)
    
    # shrunk_mat = test.get_shrunk_cov_mat()
    # print(shrunk_mat)