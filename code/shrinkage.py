"""
Let's create a class that takes in the sample covariance matrix as an input and 
can do all the operations described in appendices A and B of the paper.
The goal is to have it output our new matrix which we will use to optimize our portfolio weights.
To do that, we need to find the shrinkage intensity, which is the most involved part of the calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock_prices = pd.read_csv(r"C:\Users\jojis\OneDrive\Documents\GitHub\mf796\data\adjclosepx.csv", index_col="Date")
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

test = Covariance_Shrinkage(covariance,correlation)
F = test.get_F()
print(F)


