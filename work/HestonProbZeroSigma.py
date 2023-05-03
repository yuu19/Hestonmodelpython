import numpy as np

def HestonProbZeroSigma(phi, kappa = 10, theta = 0.07, lam, tau, K = 10, S = 7, r = 0, q = 0, Pnum = 2):
    # Log of the stock price.
    x = np.log(S)
    
    # Parameter "a" is the same for P1 and P2.
    a = kappa * theta
    
    # Parameters "u" and "b" are different for P1 and P2.
    if Pnum == 1:
        u = 0.5
        b = kappa + lam
    else:
        u = -0.5
        b = kappa + lam
    
    # D and C when sigma = 0
    D = (u * 1j * phi - phi**2 / 2) * (1 - np.exp(-b * tau)) / b
    C = (r - q) * 1j * phi * tau + a * (u * 1j * phi - 0.5 * phi**2) / b * (tau - (1 - np.exp(-b * tau)) / b)
    
    # fは(1.48)式で与えられている                                                                                                                                                                                                                                                                                                                                                                                         he characteristic function.
    f = np.exp(C + D * theta + 1j * phi * x)
    
    # (1.69)式の被積分関数
    # Return the real part of the integrand.
    y = np.real(np.exp(-1j * phi * np.log(K)) * f / 1j / phi)
    
    return y
