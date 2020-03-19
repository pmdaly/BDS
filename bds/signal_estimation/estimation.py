from collections import namedtuple

class Estimator:

    def __init__(self, model):
        self.model = model

    def estimate(self, patient):
        results = list()
        for bo in patient.brain_objects:
            #results.append(Results(bo, self.model.predict(bo)))
            results.append(self.model.predict(bo))
        return results


#Results = namedtuple('Results', ['brain_object', 'ful'])
