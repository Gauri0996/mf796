"""
Let's create a class that takes in the sample covariance matrix as an input and 
can do all the operations described in appendices A and B of the paper.
The goal is to have it output our new matrix which we will use to optimize our portfolio weights.
To do that, we need to find the shrinkage intensity, which is the most involved part of the calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

george = r"C:\Users\jojis\OneDrive\Documents\GitHub\mf796\data\adjclosepx.csv"
robbie = "/home/robbie/Documents/MSMF/Spring2020/MF796/project/data/adjclosepx.csv"
stock_prices = pd.read_csv(robbie, index_col="Date")
stock_prices.dropna(inplace=True)
returns = stock_prices.pct_change(1)
returns = returns.loc['04/01/2016':'31/12/2019',:]
covariance = returns.cov()
correlation = returns.corr()

class Covariance_Shrinkage():

    def __init__(self,cov_mat, corr_mat):
        self.cov_mat = np.array(cov_mat)
        self.corr_mat = np.array(corr_mat)

    def get_F(self):
        N = len(self.corr_mat[0:]) 
        F = np.zeros(  (N,N)   )
        r_bar_sum = 0 

        for i in range(0,N-1):
            for j in range(i+1,N):
                r_bar_sum += self.corr_mat[i,j]
        r_bar = (2/(N*(N-1)))*r_bar_sum
        
        for i in range(0,N):
            for j in range(0,N):
                if i == j:
                    F[i,j] = self.cov_mat[i,j]
                else:
                    F[i,j] = r_bar * np.sqrt(self.cov_mat[i,i]*self.cov_mat[j,j])
        return F

    def get_delta(self):
        """
        returns an estimate of the optimal shrinkage constant (intensity)
        see appendix B in honey.pdf
        """

        T = len(returns)
        N = len(returns.columns)
        pi_hat = 0
        rho_hat = 0
        gamma_hat = 1
        #pi_hat - this takes way too long to run
        for i in range(N-1):
            for j in range(N-1):
                for t in range(T-1):
                    pi_hat += ((returns.iloc[t:t+1,j:j+1].values[0][0] - returns.loc[: , returns.columns[j]].mean())*(returns.iloc[t:t+1,i:i+1].values[0][0] - returns.loc[: , returns.columns[i]].mean()) - self.cov_mat[i,j])**2
                    #print statements to make sure something is actually happening
                    print("i =", i)
                    print("j =", j)
                    print("pi_hat =", pi_hat)
                pi_hat *= 1/T
        #rho_hat and gamma_hat - to be completed

        kappa_hat = (pi_hat-rho_hat)/gamma_hat
        delta_hat = max(0,min((kappa_hat/T),1))
        return delta_hat

test = Covariance_Shrinkage(covariance,correlation)
F = test.get_F()
print(F)
test.get_delta()
