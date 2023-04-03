import numpy as np
from scipy.stats import norm
from HestonPriceZeroSigma import HestonPriceZeroSigma # heston_pricingは自分で作成したHestonモデルのプライシングコードのモジュール名

K = 100
S = 100
r = 0.03
q = 0.02
T = 0.5

kappa = 5
v0 = 0.05
theta = v0
lambda_ = 0

# Integration grid
dphi = 0.01
Uphi = 100
Lphi = 0.00001

# Black-Scholes Call and Put. Uses v0 as the variance
d1 = (np.log(S/K) + (r-q+v0/2)*T)/np.sqrt(v0*T)
d2 = d1 - np.sqrt(v0*T)
BSCall = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
BSPut = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

# Heston call and put
HCall = HestonPriceZeroSigma('C', kappa, theta, lambda_, T, K, S, r, q, Uphi, dphi, Lphi)
HPut = HestonPriceZeroSigma('P', kappa, theta, lambda_, T, K, S, r, q, Uphi, dphi, Lphi)

# Output the results
print('Black Scholes as a special case of Heston')
print(f'Using a volatility of {theta:.4f}')
print('--------------------------------------------')
print('Model               Call            Put')
print('--------------------------------------------')
print(f'Black Scholes    {BSCall:.8f}   {BSPut:.8f}')
print(f'Heston           {HCall:.8f}   {HPut:.8f}')
print('--------------------------------------------')