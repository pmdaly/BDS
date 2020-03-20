from bds.seeg_models import Patient, Population
from bds.signal_estimation import Estimator
from bds.signal_processing import WaveletReduce


patients = [Patient() for _ in range(5)]
population = Population(patients)
population.merge_elecs()
population.set_patient_models()
population.build_full_model()

est = Estimator(population.model)
p1_bos_full = est.estimate(population[0])

spectra = list()
for bo_full in p1_bos_full:
    spectra.append(WaveletReduce().get_wavelts(bo_full.get_data(), 512))
