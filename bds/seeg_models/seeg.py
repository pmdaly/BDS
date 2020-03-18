import numpy as np
import pandas as pd
import supereeg as se


class SEEGBase:

    def get_locs(self):
        return self.model.get_locs()

    def get_model(self):
        return self.model.get_model()


def random_brain_object():
    import string
    fs = 512
    n_samples = fs * 10 * 60
    n_elecs = 5
    elec_coords = ['x', 'y', 'z']
    elec_names = list(string.ascii_lowercase)[:n_elecs]
    locs = pd.DataFrame(np.random.randn(n_elecs, 3), columns=elec_coords, index=elec_names)
    data = pd.DataFrame(np.random.randn(n_samples, n_elecs), columns=elec_names)
    return se.Brain(data=data, locs=locs, sample_rate=fs)

