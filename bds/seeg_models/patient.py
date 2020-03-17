import pandas as pd
import numpy as np
import supereeg as se


class Patient:

    def __init__(self, brain_object_file='example_data'):
        self.brain_object_file = brain_object_file

    def load_data(self):
        if self.brain_object_file == 'example_data':
            self.brain_object = _gen_random_brain_object()
        else:
            self.brain_object = se.load(self.brain_object_file)

    def expand_to_population(self, population_elecs):
        pass


def _gen_random_brain_object():
    import string
    fs = 512
    n_samples = fs * 10 * 60
    n_elecs = 5
    elec_coords = ['x', 'y', 'z']
    elec_names = list(string.ascii_lowercase)[:n_elecs]
    locs = pd.DataFrame(np.random.randn(n_elecs, 3), columns=elec_coords, index=elec_names)
    data = pd.DataFrame(np.random.randn(n_samples, n_elecs), columns=elec_names)
    return se.Brain(data=data, locs=locs, sample_rate=fs)


if __name__ == '__main__':
    from .population import Population
    patients = [Patient() for _ in range(5)]
    population = Population(patients)
