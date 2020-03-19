from bds.seeg_models import Patient, Population
from bds.signal_estimation import Estimator

patients = [Patient() for _ in range(5)]
population = Population(patients)
population.merge_elecs()
population.set_patient_models()
population.build_full_model()

est = Estimator(population.model)
p1_full = est.estimate(population[0])
