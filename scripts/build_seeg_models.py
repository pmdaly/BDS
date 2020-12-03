import pandas as pd
import random
from bds.seeg_models import Patient, Population
from bds.signal_processing.wavelet import get_wavelets

random.seed(1)

outdir = '../out/'
fs = 512

patients = [Patient() for _ in range(3)]
population = Population(patients)
population.expand_patient_models()
population.build_full_model()
population.save_models(outdir)

spectra = dict()
for patient in population:
    relative_frames, absolute_frames = zip(*[get_wavelets(pbo, fs) for pbo in patient.predict_bos()])
    spectra[patient.pid] = pd.concat(wave_frames)
import ipdb; ipdb.set_trace()
