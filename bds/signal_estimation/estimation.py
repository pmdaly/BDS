from collections import namedtuple

class Estimator:

    def __init__(self, model):
        self.model = model

    def estimate(self, patient):
        results = list()
        for brain_object in patient:
            results.append(self.model.predict(brain_object))
        return results
