import pandas as pd
import numpy as np
import supereeg as se
from .seeg import SEEGBase, random_brain_object


class Patient(SEEGBase):

    patient_num = 0

    def __init__(self, pid=None, brain_object_files=None):
        self.pid = pid
        self.brain_object_files = brain_object_files
        self._load_brain_objects()
        if pid is None:
            Patient.patient_num += 1
            self.pid = f'pid{self.patient_num}'

    def _load_brain_objects(self):
        if self.brain_object_files is not None:
            self.brain_objects = [se.load(fn) for fn in self.brain_object_files]
        else:
            self.brain_object_files = ['random brain object loaded']
            self.brain_objects = [random_brain_object()]

    def build_model(self, population_elecs=None):
        if population_elecs is not None:
            model = se.Model(self.brain_objects, locs=population_elecs)
        else:
            model = se.Model(self.brain_objects)
        model.n_subs = 1
        self.model = model

    def __getitem__(self, idx):
        return self.brain_objects[idx]
