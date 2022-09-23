import numpy as np

__FACTORS = {
    'W': 52,
    'M': 12,
    'Y': 1,
    'D': 252,
    'H': 24 * 252,
    'm': 24 * 60 * 252
}


def annualization_factor(freq_type, freq=1):
    return __FACTORS[freq_type] / freq


def close_to_close(closes, freq_type='D', freq=1):
    f = annualization_factor(freq_type, freq)
    v = np.power(np.diff(np.log(closes)), 2)
    return np.sqrt(np.mean(v) * f)


def parkinson(highs, lows, freq_type='D', freq=1):
    f = annualization_factor(freq_type, freq)
    v = np.power(np.log(highs/lows), 2) / (4 * np.log(2))
    return np.sqrt(np.mean(v) * f)


def garman_klass(opens, highs, lows, closes, freq_type='D', freq=1):
    f = annualization_factor(freq_type, freq)
    v = .5 * np.power(np.log(highs / lows), 2)
    v -= (2*np.log(2)-1) * np.power(np.log(closes / opens), 2)
    return np.sqrt(np.mean(v) * f)


def rogers_satchell(opens, highs, lows, closes, freq_type='D', freq=1):
    f = annualization_factor(freq_type, freq)
    ho = np.log(highs / opens)
    co = np.log(closes / opens)
    lo = np.log(lows / opens)
    v = ho * (ho - co) + lo * (lo - co)
    return np.sqrt(np.mean(v) * f)


def garman_klass_yang_zhang(opens, highs, lows, closes, prev_closes,
                            freq_type='D', freq=1):
    f = annualization_factor(freq_type, freq)
    v = .5 * np.power(np.log(highs / lows), 2)
    v += np.power(np.log(opens / prev_closes), 2)
    v -= (2 * np.log(2) - 1) * np.power(np.log(closes / opens), 2)
    return np.sqrt(np.mean(v) * f)


def yang_zhang(opens, highs, lows, closes, prev_closes,
               freq_type='D', freq=1):
    f = annualization_factor(freq_type, freq)
    n = closes.shape[0]
    k = .34 / (1.34 + (n+1)/(n-1))
    rs = rogers_satchell(opens, highs, lows, closes, freq_type, freq)
    rs = np.power(rs, 2)
    oc = np.log(opens / prev_closes)
    co = np.log(closes / opens)
    overnight_var = np.sum(np.power(oc - np.mean(oc), 2)) / (n-1)
    openclose_var = np.sum(np.power(co - np.mean(co), 2)) / (n-1)
    v = (1-k) * rs + (overnight_var + k * openclose_var) * f
    return np.sqrt(v)


cc = close_to_close
gk = garman_klass
gkyz = garman_klass_yang_zhang
pa = parkinson
pk = parkinson
rs = rogers_satchell
yz = yang_zhang
