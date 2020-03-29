"""
Let's create a class that takes in the sample covariance matrix as an input and 
can do all the operations described in appendices A and B of the paper.
The goal is to have it output our new matrix which we will use to optimize our portfolio weights.
To do that, we need to find the shrinkage intensity, which is the most involved part of the calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" Change the path to your path if you are gonna run this """
george = r"C:\Users\jojis\OneDrive\Documents\GitHub\mf796\data\adjclosepx.csv"
robbie = "/home/robbie/Documents/MSMF/Spring2020/MF796/project/data/adjclosepx.csv"
issy = "/Users/issyanand/Desktop/adjclosepx.csv"
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
        self.N = len(corr_mat[0:])

    def get_F(self):

        F = np.zeros(  (self.N,self.N)   )
        r_bar_sum = 0 

        for i in range(0,self.N-1):
            for j in range(i+1,self.N):
                r_bar_sum += self.corr_mat[i,j]
        r_bar = (2/(self.N*(self.N-1)))*r_bar_sum
        
        for i in range(0,self.N):
            for j in range(0,self.N):
                if i == j:
                    F[i,j] = self.cov_mat[i,j]
                else:
                    F[i,j] = r_bar * np.sqrt(self.cov_mat[i,i]*self.cov_mat[j,j])
        return F
    
    def get_gamma(self):
        gamma = 0
        F = self.get_F()
        for i in range(0,self.N):
            for j in range(0,self.N):
                gamma += ( (F[i,j] - self.cov_mat[i,j])**2 )
        return gamma

    def get_pi(self):
        #this takes way too long to run
        # ^^^ passing the returns either as an argument to this function or in the init might be a good idea (that way T can also be called with self)
        pi = 0
        T = len(returns)
        #in all the loops below, I think we need to remove the -1. The range function is not inclusive for the last number (i.e. range(3) = [0,1,2])
        for i in range(0,self.N-1):
            for j in range(0,self.N-1):
                for t in range(0,T-1):
                    pi += ((returns.iloc[t:t+1,i:i+1].values[0][0] - returns.loc[: , returns.columns[i]].mean())*(returns.iloc[t:t+1,j:j+1].values[0][0] - returns.loc[: , returns.columns[j]].mean()) - self.cov_mat[i,j])**2
                    #print statements to make sure something is actually happening
                    #print("i =", i)
                    #print("j =", j)
                    #print("pi =", pi)
                pi *= 1/T
        return pi
    
    def get_rho(self):
        #going to need to pass the same improvements for the pi function through this as otherwise it will also take ages
        
        T = len(returns)
        
        r_bar_sum = 0 

        for i in range(0,self.N-1):
            for j in range(i+1,self.N):
                r_bar_sum += self.corr_mat[i,j]
        r_bar = (2/(self.N*(self.N-1)))*r_bar_sum
        
        sum1 = 0
        for i in range(self.N):
            for j in range(self.N):
                if (j==i)==False:
                    theta_ii = 0
                    for t in range(0,T-1):
                        comp_1 = ((returns.iloc[t:t+1,i:i+1].values[0][0] - returns.loc[: , returns.columns[i]].mean())**2 - self.cov_mat[i,i]) 
                        comp_2 = ((returns.iloc[t:t+1,i:i+1].values[0][0] - returns.loc[: , returns.columns[i]].mean())*(returns.iloc[t:t+1,j:j+1].values[0][0] - returns.loc[: , returns.columns[j]].mean()) - self.cov_mat[i,j])
                        theta_ii += comp_1*comp_2
                    theta_ii *= 1/T
        
                    theta_jj = 0
                    for t in range(0,T-1):
                        comp_1 = ((returns.iloc[t:t+1,j:j+1].values[0][0] - returns.loc[: , returns.columns[j]].mean())**2 - self.cov_mat[j,j]) 
                        comp_2 = ((returns.iloc[t:t+1,i:i+1].values[0][0] - returns.loc[: , returns.columns[i]].mean())*(returns.iloc[t:t+1,j:j+1].values[0][0] - returns.loc[: , returns.columns[j]].mean()) - self.cov_mat[i,j])
                        theta_jj += comp_1*comp_2
                    theta_jj *= 1/T
                    
                    sum1 += r_bar/2*(np.sqrt(self.cov_mat[j][j]/self.cov_mat[i][i])*theta_ii + np.sqrt(self.cov_mat[i][i]/self.cov_mat[j][j])*theta_jj)
        
        sum2 = 0
        for i in range(self.N):
            for t in range(0,T-1):
                    sum2 += ((returns.iloc[t:t+1,i:i+1].values[0][0] - returns.loc[: , returns.columns[i]].mean())**2 - self.cov_mat[i,i])**2
            sum2 *= 1/T
        
        rho = sum1 + sum2
        return rho
    
    def get_delta(self):
        
        T = len(returns)
        kappa = (self.get_pi() - self.get_rho())/self.get_gamma()
        delta = max(0, min(kappa/T,1))
        return delta

#making sure things are working
test = Covariance_Shrinkage(covariance,correlation)
F = test.get_F()
gamma = test.get_gamma()
#pi = test.get_pi()
print(F)
print(gamma)
#print(pi)

