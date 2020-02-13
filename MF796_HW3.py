#Robbie Cook
#rjcook@bu.edu
#MF796 - Problem set #3
#February 12, 2020

import numpy
import math
import cmath
from scipy.stats import norm
from matplotlib import pyplot as plt

#General StochasticProcess base class
class StochasticProcess:
    def __init__(self, maturity=0.5, initial_price=250.0, strike=250.0, risk_free_rate=0.02, dividend=0.0):
        self.T = maturity
        self.S = initial_price
        self.K = strike
        self.r = risk_free_rate
        self.q = dividend
        
    def chacteristic_function(self,u=1):
        print("method not defined for base class")
        
    def price(self):
        print("method not defined for base class")
        
#Black Scholes formula class
class BlackScholesProcess(StochasticProcess):
    def __init__(self, volatility=0.12, maturity=0.25, initial_price=150.0, strike=150.0, risk_free_rate=0.025, dividend=0.0):
        self.sigma = volatility
        StochasticProcess.__init__(self, maturity, initial_price, strike, risk_free_rate, dividend)
    
    def price(self):
        sigmaRtT = self.sigma * math.sqrt(self.T)
        rSigTerm = (self.r + ((self.sigma**2)/2.0)) * self.T
        d1 = (math.log(self.S/self.K) + rSigTerm) / sigmaRtT
        d2 = d1 - sigmaRtT
        term1 = self.S * norm.cdf(d1)
        term2 = self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        return term1 - term2
    
    def vega(self):
        sigmaRtT = self.sigma * math.sqrt(self.T)
        rSigTerm = (self.r + ((self.sigma**2)/2.0)) * self.T
        d1 = (math.log(self.S/self.K) + rSigTerm) / sigmaRtT
        phi_d1 = (math.exp(-(d1**2/2)))/(math.sqrt(2*math.pi))
        return self.S*phi_d1*math.sqrt(self.T)
        
#Heston class
class HestonProcess(StochasticProcess):
    def __init__(self, vol_of_vol=0.2, initial_vol=0.08, vol_mean_reversion=0.7, bm_correlation=-0.4, vol_mean=0.1, maturity=0.5, initial_price=250, strike=250, risk_free_rate=0.02, dividend=0):
        #Heston parameters
        
        self.little_sigma = vol_of_vol
        self.v0 = initial_vol
        self.kappa = vol_mean_reversion
        self.rho = bm_correlation
        self.big_sigma = vol_mean
        StochasticProcess.__init__(self, maturity, initial_price, strike, risk_free_rate, dividend)
    
    def characteristic_function(self, u):
        #Known characteristic function for Heston model
        
        first_complex_term = u**2 + u*1j
        second_complex_term = (self.kappa - 1j*self.rho*self.little_sigma*u)**2
        lambd = cmath.sqrt(self.little_sigma**2 * first_complex_term + second_complex_term)
        
        numerator_term = (self.kappa*self.big_sigma*self.T*(self.kappa-1j*self.rho*self.little_sigma*u))/self.little_sigma**2
        numerator = cmath.exp(1j*u*math.log(self.S) + 1j*u*(self.r-self.q)*self.T + numerator_term)
        denominator_term = (self.kappa-1j*self.rho*self.little_sigma*u)/lambd
        denominator = (cmath.cosh((lambd*self.T)/2) + denominator_term*cmath.sinh((lambd*self.T)/2))**((2*self.kappa*self.big_sigma)/self.little_sigma**2)
        w = numerator/denominator
        
        char_numerator = -(u**2 + 1j*u)*self.v0
        coth_term = lambd*(cmath.cosh((lambd*self.T)/2)/(cmath.sinh((lambd*self.T)/2)))
        char_denominator = coth_term + self.kappa - 1j*self.rho*self.little_sigma*u
        char_function = w*cmath.exp(char_numerator/char_denominator)
    
        return char_function
    
    def price(self, fft):
        #price function via FFT
        #returns the price of the at-the-money call
        
        prices =  self.price_vector(fft)
        strikes = self.strike_grid(fft)
        #find the index of the at-the-money strike
        for m in range(len(strikes)):
            if strikes[m] < self.K+1 and strikes[m] > self.K-1:
                target_k = m
        #return the price that maps to the at-the-money strike
        return prices[target_k]
    
    def price_vector(self, fft):
        #returns the price vector
        
        return fft.fft(self)
    
    def strike_grid(self, fft):
        #returns the strike grid
        
        prices = fft.fft(self)
        beta = math.log(self.S) - (fft.delta_k*fft.N)/2
        k = numpy.zeros(fft.N)
        K = numpy.zeros(fft.N)
        target_k = 0
        #define the strike grid
        for m in range(fft.N):
            k[m] = beta + (m)*fft.delta_k
            K[m] = math.exp(k[m])
        return K
    
    def implied_vol(self, call_price, sigma):
        #calculate implied volatility using Black-Scholes and Newton's method
        epsilon = 0.05
        black_scholes_call = BlackScholesProcess(volatility=sigma, maturity=self.T, initial_price=self.S, strike=self.K, risk_free_rate=self.q, dividend=self.q)
        if abs(black_scholes_call.price() - call_price) < epsilon:
            return sigma
        else: 
            new_sigma = sigma - ((black_scholes_call.price() - call_price)/black_scholes_call.vega())
            return self.implied_vol(call_price, new_sigma)
    
#FFT class
class FFT:
    def __init__(self, damping_factor=1.5, number_of_nodes=2**10, upper_bound=750):
        self.alpha = damping_factor
        self.N = number_of_nodes
        self.B = upper_bound
        self.delta_v = self.B/self.N
        self.delta_k = (2*math.pi)/(self.delta_v*self.N)
        
    def fft(self, process):
        x = numpy.zeros(self.N, dtype=complex)
        big_delta = 0
        
        #build input vector x
        for i in range(1,len(x)+1):
            if i == 1:
                big_delta = 1
            fft_numerator = ((2-big_delta)*self.delta_v)*math.exp(-process.r)
            fft_denominator = 2*(self.alpha + 1j*((i-1)*self.delta_v))*(self.alpha + 1j*((i-1)*self.delta_v) + 1)
            fft_exp = cmath.exp(-1j*(math.log(process.S)-((self.delta_k*self.N)/2))*(i-1)*self.delta_v)
            char_func_input = ((i-1)*self.delta_v)-(self.alpha+1)*1j
            x[i-1] = (fft_numerator/fft_denominator)*fft_exp*process.characteristic_function(char_func_input)
            big_delta = 0
            
        #pass x into fft algorithm
        y = numpy.fft.fft(x)
        
        #translate output vector to call prices
        prices = numpy.zeros(self.N)
        for i in range(len(y)):
            prices[i] = ((math.exp(-self.alpha*(math.log(process.S)-self.delta_k*(self.N/2-(i-1)))))/math.pi)*y[i].real
        return prices


# In[108]:


#Problem 1 part a-i
heston = HestonProcess()
fft = FFT(damping_factor=1.5)
heston.price(fft)


# In[109]:


#Problem 1 part a-ii
fft = FFT(number_of_nodes=2**10,upper_bound=750)
heston.price(fft)


# In[110]:


#Problem 1 part a-iii
heston = HestonProcess(strike=260)
fft = FFT(number_of_nodes=2**10,upper_bound=750)
heston.price(fft)


# In[113]:


#Problem 1 part b-i
strike_targets = [115.0,120.0,125.0,130.0,135.0,140.0,145.0,150.0,155.0,160.0,165.0,170.0]
fft = FFT(number_of_nodes=2**13)
heston_prices = numpy.zeros(len(strike_targets))
implied_vols = numpy.zeros(len(strike_targets))
for i in range(len(strike_targets)):
    heston = HestonProcess(vol_of_vol=0.4, initial_vol=0.09, vol_mean_reversion=0.5, bm_correlation=0.25, vol_mean=0.12, maturity=0.25, initial_price=150, strike=strike_targets[i], risk_free_rate=0.025)
    heston_prices[i] = heston.price(fft)
    initial_v0_guess = math.sqrt((2*math.pi)/heston.T)*(heston_prices[i]/heston.S)
    implied_vols[i] = heston.implied_vol(heston_prices[i],initial_v0_guess)
    print("strike = ", strike_targets[i])
    print("price = ", heston.price(fft))
    print("implied vol = ", implied_vols[i])
    print("--------------")
    

# In[114]:


#Problem 1 part b-ii
maturity_targets = [0.083,0.167,0.25,0.5,1.0,2.0,3.0,4.0]
fft = FFT(number_of_nodes=2**10)
heston_prices = numpy.zeros(len(maturity_targets))
implied_vols = numpy.zeros(len(maturity_targets))
for i in range(len(maturity_targets)):
    heston = HestonProcess(vol_of_vol=0.4, initial_vol=0.09, vol_mean_reversion=0.5, bm_correlation=0.25, vol_mean=0.12, maturity=maturity_targets[i], initial_price=150, strike=150, risk_free_rate=0.025)
    heston_prices[i] = heston.price(fft)
    initial_v0_guess = math.sqrt((2*math.pi)/heston.T)*(heston_prices[i]/heston.S)
    implied_vols[i] = heston.implied_vol(heston_prices[i],initial_v0_guess)
    print("maturity = ", maturity_targets[i])
    print("price = ", heston.price(fft))
    print("implied vol = ", implied_vols[i])
    print("--------------")


# In[297]:


#Problem 1 part b-iii
#parameters - sigma, v0, kappa, rho, Big_sigma, r
#volatility skew - plot implied vol (y) vs strike (x)
#term structure - plot implied vol (y) vs maturity (x)
