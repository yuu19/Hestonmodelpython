import numpy as np
from scipy.integrate import trapz
from HestonProbZeroSigma import HestonProbZeroSigma
def HestonPriceZeroSigma(PutCall, kappa, theta, lambd, tau, K, S, r, q, Uphi, dphi, Lphi):
    
    # Get the integrands for the two Heston ITM probabilities p1 and p2
    phi = np.arange(Lphi, Uphi + dphi, dphi)
    P1_int = np.array([HestonProbZeroSigma(x, kappa, theta, lambd, tau, K, S, r, q, 1) for x in phi])
    P2_int = np.array([HestonProbZeroSigma(x, kappa, theta, lambd, tau, K, S, r, q, 2) for x in phi])
    
    # Integrate to get p1 and p2
    p1 = 0.5 + (1 / np.pi) * trapz(P1_int, phi)
    p2 = 0.5 + (1 / np.pi) * trapz(P2_int, phi)
    
    # Restrict p1 and p2 to the interval [0,1]
    p1 = max(min(1, p1), 0)
    p2 = max(min(1, p2), 0)
    
    # Heston Call price directly.
    HestonC = S * p1 * np.exp(-q * tau) - K * np.exp(-r * tau) * p2
    
    # Heston put price by put-call parity;
    HestonP = HestonC + K * np.exp(-r * tau) - S * np.exp(-q * tau)
    
    if PutCall == 'C':
        y = HestonC
    else:
        y = HestonP
    
    return y
