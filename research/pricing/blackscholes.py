import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton


ln = np.log
exp = np.exp
pow = np.power
sqrt = np.sqrt


def bsb_approx(T, S, sigma):
    """
    Brenner Subramanian approximation
    Applicability: sigma*sqrt(T) should be small and K ATM
    """
    
    price = .04 * S * sigma * sqrt(T)
    delta = .5 + .2 * sigma * sqrt(T)
    vega = .4 * sqrt(T) * S
    return dict(
        price=price,
        delta=delta,
        vega=vega
    )


def dplus(t, S, T, K, r, sigma, q):
    """Aussi dénoté d+ ou d1"""
    tau = T-t
    ss = sigma * sigma
    return (ln(S/K) + (r - q + .5 * ss) * tau) / (sigma * sqrt(tau))


def dminus(t, S, T, K, r, sigma, q):
    """Aussi dénoté d- ou d2"""
    return dplus(t, S, T, K, r, sigma, q) - sigma * sqrt(T-t)


def call_parity(t, S, T, K, r, q, put):
    """Computes the Call given a put price using put call parity"""
    tau = T - t
    ract = exp(-r*tau)
    qact = exp(-q*tau)
    return put + S * qact - K * ract


def put_parity(t, S, T, K, r, q, C):
    tau = T - t
    ract = exp(-r*tau)
    qact = exp(-q*tau)
    return (C - (S * qact) + (K * ract))


def bachelier(t, S, T, K, r, sigma):
    tau = T - t
    SK = S-K
    sigsqt = sigma * sqrt(tau)
    N = norm.cdf
    D = norm.pdf
    C = SK * N(SK/sigsqt) + sigsqt * D(SK/sigsqt)
    return


def binary(t, S, T, K, r, sigma, q=0):
    tau = T - t
    sqt = sqrt(tau)
    ract = exp(-r*tau)
    qact = exp(-q*tau)
    ss = sigma * sigma
    d1 = (np.log(S/K) + (r - q + .5 * ss) * tau) / (sigma * sqt)
    d2 = d1 - sigma * sqt

    N = norm.cdf
    D = norm.pdf

    C = ract * N(d2)
    P = ract * N(-d2)
    delta_call = ract * D(d2) / (sigma * sqt * S * qact)
    delta_put = - ract * D(-d2) / (S * sigma * sqt * qact)
    delta_call_k = - ract / (K * sigma * sqt) * D(d2)
    return dict(
        C=C,
        P=P,
        delta_call=delta_call,
        delta_call_k=delta_call_k,
        delta_put=delta_put,
    )


def bsm(t, S, T, K, r, sigma, q=0):
    tau = T - t
    sqt = sqrt(tau)
    ract = exp(-r*tau)
    qact = exp(-q*tau)
    ss = sigma * sigma

    d1 = dplus(t, S, T, K, r, sigma, q)
    d2 = dminus(t, S, T, K, r, sigma, q)

    N = norm.cdf
    D = norm.pdf

    C = S * qact * N(d1) - K * ract * N(d2)
    P = ract * K * N(-d2) - S * qact * N(-d1)

    delta_call = qact * N(d1)
    delta_put = -qact * N(-d1)

    delta_call_k = - ract * N(d2)
    delta_put_k = ract * N(-d2)

    gamma_k = ract * D(d2) / (K * sigma * sqrt(tau))

    vega = S * qact * D(d1) * sqt

    theta_call = -(
        -qact * S * D(d1) * sigma /
        (2*sqt) - r * K * ract * N(d2) + q * S * qact * N(d1)
    )
    theta_put = -(
        -qact * S * D(d1) * sigma /
        (2 * sqt) + r * K * ract * N(-d2) - q * S * qact * N(-d1)
    )

    rho_call = K * tau * ract * N(d2)
    rho_put = -K * tau * ract * N(-d2)

    gamma = qact * D(d1) / (S * sigma * sqt)

    vanna = vega * (1 - d1/(sigma * sqt)) / S

    vomma = vega * d1 * d2 / sigma
    
    return dict(
        nd1=N(d1),
        nd2=N(d2),
        C=C,
        P=P,
        delta_call=delta_call,
        delta_put=delta_put,
        delta_call_k=delta_call_k,
        delta_put_k=delta_put_k,
        gamma=gamma,
        gamma_k=gamma_k,
        vega=vega,
        theta_call=theta_call,
        theta_put=theta_put,
        rho_call=rho_call,
        rho_put=rho_put,
        vanna=vanna,
        vomma=vomma
    )


def B(t, T, r):
    return exp(-r*(T-t))


def forward(t, T, r, C):
    return C / B(t, T, r)

