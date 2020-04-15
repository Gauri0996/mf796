import numpy as np
import pandas as pd

class Covariance_Shrinkage:
    def __init__(self,stock_returns):
        self.stock_returns = stock_returns
        self.t = len(stock_returns)
        self.n = len(stock_returns.iloc[0,:])
        self.meanx = stock_returns.mean()
        self.x = stock_returns - self.meanx

    def get_sample_cov_mat(self):
        sample = (1/self.t)*(self.x.transpose() @ self.x)
        return sample
    
    def get_shrunk_cov_mat(self):
        #prior (F)
        sample = self.get_sample_cov_mat()
        var = np.diag(sample)
        sqrtvar = np.sqrt(var)
        #getting rBar
        a = np.array(self.n*[list(sqrtvar)]).transpose()
        b = sample / (a*a.transpose())
        c = np.sum(b)
        d = np.sum(c) - self.n
        rBar = d/(self.n*(self.n-1))
        #getting F
        prior = rBar*(a*a.transpose())
        np.fill_diagonal(prior,var)


        #pi-hat
        y = self.x**2
        phiMat = y.transpose()@y/self.t - 2*(self.x.transpose()@self.x)*sample/self.t + sample**2
        phi = np.sum(np.sum(phiMat))

        #rho-hat
        term1 = ((self.x**3).transpose() @ self.x) / self.t
        help_mat = self.x.transpose() @ self.x/self.t
        helpDiag = np.diag(help_mat)
        term2 = np.array(self.n*[list(helpDiag)]).transpose() * sample
        term3 = help_mat * np.array(self.n*[list(var)]).transpose()
        term4 = np.array(self.n*[list(var)]).transpose() * sample
        thetaMat = np.array(term1 - term2 - term3 +term4)
        np.fill_diagonal(thetaMat,0)
        sqrtvar1 = pd.DataFrame(sqrtvar)
        rho = np.sum(np.diag(phiMat)) + rBar * np.sum(np.sum( ((1/sqrtvar1) @ sqrtvar1.transpose()) *thetaMat ))

        #gamma-hat
        gamma = np.linalg.norm(sample-prior,'fro')**2

        #shrinkage constant
        kappa = (phi-rho)/gamma
        shrinkage = max(0,min(1,kappa/self.t))
        sigma = shrinkage*prior + (1-shrinkage)*sample
        return sigma

if __name__=="__main__":
    """ Change the path to your path if you are gonna run this """

    george = r"C:\Users\jojis\OneDrive\Documents\GitHub\mf796\data\adjclosepx.csv"
    robbie = "/home/robbie/Documents/MSMF/Spring2020/MF796/project/data/adjclosepx.csv"
    issy = "/Users/issyanand/Desktop/adjclosepx.csv"
    stock_prices = pd.read_csv(george, index_col="Date")
    stock_prices.dropna(inplace=True)
    stock_returns = stock_prices.pct_change(1)
    stock_returns = stock_returns.loc['02/01/2015':'31/12/2019',:]

    test = Covariance_Shrinkage(stock_returns)
    shrunk_mat = test.get_shrunk_cov_mat()
    cov = test.get_sample_cov_mat()
    print(shrunk_mat - cov)