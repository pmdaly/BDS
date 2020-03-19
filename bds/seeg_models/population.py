import pandas as pd
import supereeg as se
from .seeg import SEEGBase

class Population(SEEGBase):

    def __init__(self, patients):
        self.patients = patients

    def merge_elecs(self):
        locs = list()
        for patient in self.patients:
            for bo in patient.brain_objects:
                locs.append(bo.locs)
        self.locs = pd.concat(locs)

    def set_patient_models(self):
        for patient in self.patients:
            patient.build_model(self.locs)

    def build_full_model(self):
        self.model = se.Model([pt.model for pt in self.patients])

    def __getitem__(self, idx):
        return self.patients[idx]
