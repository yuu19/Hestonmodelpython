import numpy as np
# Returns the integrand for the risk neutral probabilities P1 and P2.
# phi = integration variable
# Pnum = 1 or 2 (for the probabilities)
# Heston parameters:
#    kappa  = volatility mean reversion speed parameter
#    theta  = volatility mean reversion level parameter
#    lambda = risk parameter
#    rho    = correlation between two Brownian motions
#    sigma  = volatility of variance
#    v      = initial variance
# Option features.
#    PutCall = 'C'all or 'P'ut
#    K = strike price
#    S = spot price
#    r = risk free rate
#    q = dividend yield
#    Trap = 1 "Little Trap" formulation 
#           0  Original Heston formulation

def HestonProbZeroSigma(phi,kappa,theta,lam,rho,sigma,tau,K,S,r,q,v0,Pnum,Trap):
#def HestonProbZeroSigma(phi, kappa = 10, theta = 0.07, lam, tau, K = 10, S = 7, r = 0, q = 0, Pnum = 2):
  # Log of the stock price.
  x = np.log(S)
    
  # Parameter "a" is the same for P1 and P2.
  a = kappa * theta

  # Parameters "u" and "b" are different for P1 and P2.
  if Pnum==1:
    u = 0.5
    b = kappa + lam - rho*sigma
  else:
    u = -0.5
    b = kappa + lam
    
  d = np.sqrt((rho*sigma*1j*phi - b)^2 - sigma^2*(2*u*1j*phi - phi^2))
  g = (b - rho*sigma*1j*phi + d) / (b - rho*sigma*1j*phi - d)
    
  if Trap==1:
	# "Little Heston Trap" formulation
    c = 1/g
    D = (b - rho*sigma*1j*phi - d)/sigma^2*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)))
    G = (1 - c*np.exp(-d*tau))/(1-c)
    C = (r-q)*1j*phi*tau + a/sigma**2 *((b - rho*sigma*1j*phi - d)*tau - 2*np.log(G))
  elif Trap==0:
	# Original Heston formulation.
    G = (1 - g*np.exp(d*tau))/(1-g)
    C = (r-q)*1j*phi*tau + a/sigma**2 *((b - rho*sigma*1j*phi + d)*tau - 2*np.log(G))
    D = (b - rho*sigma*1j*phi + d)/sigma**2 *((1-np.exp(d*tau))/(1-g*np.exp(d*tau)))

  # The characteristic function.
  f = np.exp(C + D*v0 + 1j *phi* x)
  # Return the real part of the integrand.
  y = np.real(np.exp(-1j * phi * np.log(K)) * f / 1j / phi)
  return y







 
