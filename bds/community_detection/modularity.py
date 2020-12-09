import supereeg as se


''' community detection

TODO: Patrick
copy contents of folders in
DIR_AAL2 etc into repo


TODO: Ankit
1. Removing unneccesary code from network_modularity.py
2. Review code to ensure figures or unneccesary intermediary steps are not saved
(mainly plots)
3. Fill out  modularity class
4. Write a simple script that shows how this works, eg.

modularity = Modularity(...)
modularity.gamma_search(...)
modularity.atlas_correlation(...)
modularity.groupwise_comparission(...)

args:
    mdd list
    non-mdd list
    brain objects?
    full model?
    patient model?
    output modularity dir

returns/outs:
    community assgnments
'''

class Modularity:

    def __init__(self, pop_model_fn, mdd_only_fn, control_only_fn, atlas):
        self.pop_model = se.load(pop_model_fn)
        self.mdd_model = se.load(mdd_only_fn)
        self.control_model = se.load(control_only_fn)

    def gamma_search(self):
        '''
        1. take a pop model
        2. range of gammas
        3. apply com det for n times
        4. cache outputs
        5. construct module allegiance per gamma
        6. run com det on module allegiance
        7. output community assignments for each gamma val (property...
        self.assignments)
        '''
        pass

    def atlas_correlation(self):
        '''
        1. take assignment table, run zrand from ncat.py and output a rand sim
        stat per gamma
        2. return gamma and community assignments that are most sim to atlas (cach
        it probably)
        3. write gamma / comm assignments
        '''
        pass

    def groupwise_comparission(self):
        '''
        1. take optimal com assignments from atlas correlations and project the
        community assignments onto the mdd and control models
        2. evaluate inter/intra com correlations per group and compare
        distributions between the groups
        3. outputs a community x community matrix of effect size for connectivity
        comparrisions from one group to the other
        '''
        pass

