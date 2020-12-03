import os
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
        self.elec_names = self[0].get_data().columns.values
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
            expanded_model = se.Model(self.brain_objects, locs=population_elecs)
            expanded_model.n_subs = 1
            self.expanded_model = expanded_model
        else:
            model = se.Model(self.brain_objects)
            model.n_subs = 1
            self.model = model

    def predict_bos(self, bos=None):
        if bos:
            return [self.expanded_model.predict(bo) for bo in bos]
        else:
            return [self.expanded_model.predict(bo) for bo in self]

    def save_model(self, outdir='./', expanded=False, overwrite=False):
        if expanded:
            fn = f'{outdir}/{self.pid}_expanded.mo'
            if os.path.isfile(fn) and not overwrite: return
            self.expanded_model.save(fn)
        else:
            fn = f'{outdir}/{self.pid}.mo'
            if os.path.isfile(fn) and not overwrite: return
            self.model.save(fn)

    def __getitem__(self, idx):
        return self.brain_objects[idx]
