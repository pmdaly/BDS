import pandas as pd


class Population:

    def __init__(self, patients):
        self.patients = patients

    def merge_elecs(self):
        all_locs = pd.concat([pt.brain_object.locs for pt in self.patients])

    def load_patient_data(self):
        for patient in self.patients:
            patient.load_data()


    def expand_patients(self):
        pass

