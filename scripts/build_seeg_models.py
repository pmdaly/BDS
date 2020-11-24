from bds.seeg_models import Patient, Population
from bds.signal_estimation import Estimator
from bds.signal_processing.wavelet import get_wavelets


patients = [Patient() for _ in range(5)]
population = Population(patients)
population.merge_elecs()
population.set_patient_models()
population.build_full_model()

est = Estimator(population.model)

spectra = list()
for patient in population:
    brain_objects_full = est.estimate(patient)
    for brain_object_full in brain_objects_full:
        spectra.append(get_wavelets(brain_object_full.get_data(), 512))
