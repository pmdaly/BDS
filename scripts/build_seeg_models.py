import pandas as pd
import pickle
import random
from bds.seeg_models import Patient, Population
from bds.signal_processing.wavelet import get_wavelets


'''
mdd only, control only
'''

random.seed(1)

outdir = '../out/'
fs = 512

patients = [Patient() for _ in range(3)]
population = Population(patients)
population.expand_patient_models()
population.build_full_model()
population.save_models(outdir)

rdict = dict()
adict = dict()
for patient in population:
    relative_frames, absolute_frames = zip(*[get_wavelets(pbo, fs) for pbo in patient.predict_bos()])
    rdict[patient.pid] = pd.concat(relative_frames)
    adict[patient.pid] = pd.concat(absolute_frames)

rf = pd.concat(rdict.values())
rf.index = rdict.keys()

af = pd.concat(adict.values())
af.index = adict.keys()

with open(outdir + 'spectra.pkl', 'wb') as f:
    pickle.dump({'relative': rf, 'absolute': rf}, f)


