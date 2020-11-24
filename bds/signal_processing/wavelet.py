import numpy as np
import pywt
from pathos.multiprocessing import Pool


def get_wavelets(df, fs, n_jobs=20):
    try:
        signals = df.values
    except AttributeError:
        signals = df
    num_signals = signals.shape[1]
    power_envs = None
    with Pool(n_jobs) as p:
        pool_items = ({'elec': signals[:, elec], 'fs': fs} for elec in range(num_signals))
        power_envs = np.array(list(p.imap(_process_elec, pool_items)))
    return power_envs


def _process_elec(args):
    elec_data = args['elec']
    fs = args['fs']
    elec_envs = list()
    scales = np.logspace(np.log10(1), np.log10(150), 25)
    f = pywt.scale2frequency(wavelet='morl', scale=scales) * fs
    for sig in chunker(elec_data, 30*fs):
        coefs, freqs = pywt.cwt(sig, f, 'morl', sampling_period=1/fs)
        amps = np.abs(coefs)
        delta   = amps[np.where((freqs >=  1) & (freqs <  4))[0], :].mean()
        theta   = amps[np.where((freqs >=  4) & (freqs <  9))[0], :].mean()
        alpha   = amps[np.where((freqs >=  9) & (freqs < 13))[0], :].mean()
        beta    = amps[np.where((freqs >= 13) & (freqs < 31))[0], :].mean()
        gamma_l = amps[np.where((freqs >= 31) & (freqs < 81))[0], :].mean()
        gamma_h = amps[np.where(freqs >= 81)[0], :].mean()
        elec_envs.append([delta, theta, alpha, beta, gamma_l, gamma_h])
    return elec_envs


def chunker(iterable, chunk_size, dim=0):
    '''Returns a generator with chunk_size chunks'''
    if dim == 0:
        for chunk in range(0, len(iterable), chunk_size):
            yield iterable[chunk: chunk + chunk_size]
    else:
        for chunk in range(0, len(iterable[0]), chunk_size):
            yield iterable[:, chunk: chunk + chunk_size]
