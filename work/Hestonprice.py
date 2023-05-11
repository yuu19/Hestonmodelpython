import numpy as np
from scipy.integrate import trapz #台形近似 https://showa-yojyo.github.io/notebook/python-scipy/integrate.html
#from HestonProbZeroSigma import HestonProbZeroSigma
def HestonPrice(PutCall,kappa,theta,lam,rho,sigma,T,K,S,r,q,v0,trap,Lphi,Uphi,dphi):
    
    # Get the integrands for the two Heston ITM probabilities p1 and p2
    phi = np.arange(Lphi, Uphi + dphi, dphi)
    P1_int = np.array([HestonProb(x,kappa,theta,lam,rho,sigma,T,K,S,r,q,v0,1,trap) for x in phi])
    P2_int = np.array([HestonProb(x,kappa,theta,lam,rho,sigma,T,K,S,r,q,v0,2,trap) for x in phi])
    
    # Integrate to get p1 and p2
    p1 = 0.5 + (1 / np.pi) * trapz(P1_int, phi)
    p2 = 0.5 + (1 / np.pi) * trapz(P2_int, phi)
    
    # Heston Call price directly.
    HestonC = S * p1 * np.exp(-q * tau) - K * np.exp(-r * tau) * p2
    
    # Heston put price by put-call parity;
    HestonP = HestonC + K * np.exp(-r * tau) - S * np.exp(-q * tau)
    
    if PutCall == 'C':
        y = HestonC
    else:
        y = HestonP
    
    return y
