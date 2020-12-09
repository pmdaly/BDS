import os
import pandas as pd
import supereeg as se
from .seeg import SEEGBase

class Population(SEEGBase):

    def __init__(self, patients):
        self.patients = patients
        self._merge_elecs()

    def _merge_elecs(self):
        locs = list()
        for patient in self.patients:
            for bo in patient.brain_objects:
                locs.append(bo.locs)
        self.locs = pd.concat(locs)

    def expand_patient_models(self):
        for patient in self.patients:
            patient.build_model(self.locs)

    def build_full_model(self):
        self.model = se.Model([pt.expanded_model for pt in self.patients])

    def save_models(self, outdir='./', overwrite=False):
        for patient in self.patients:
            patient.save_model(outdir, expanded=True)
        pop_fn =f'{outdir}/population.mo'
        if os.path.isfile(pop_fn) and not overwrite: return
        self.model.save(pop_fn)

    def build_mdd_and_control_models():
        pass

    def __getitem__(self, idx):
        return self.patients[idx]
