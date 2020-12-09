import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(zscore=True):
    infile = '/userdata/pdaly/supereeg/results/samp100_2h_54pid_pos-1.18.power.pkl'
    data = pickle.load(open(infile, 'rb'))
    labels = pd.read_csv('/home/kscangos/Sandbox/full_patient_list_pd_feb.csv', index_col=0)
    labels['pid'] = labels.index.map(lambda pid: pid[2:])
    df = data['relative']
    df = df.reset_index().merge(labels[['pid', 'dep']], left_on='index',
                           right_on='pid').set_index('index').drop('pid', axis=1)
    df.rename(columns={'Dep': 'dep'}, inplace=True)
    df.drop(['92','111','119','131','135','27','115','130',
             '144','158','162','170','183'], axis=0, inplace=True)
    X, y = df.drop('dep', axis=1), df.dep
    if zscore:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X, y


class Modeler:

    def __init__(self, estimator, params, metrics, pca=False):
        self.estimator = estimator
        self.params = params
        self.metrics = metrics
        self.pca = pca
        self.steps = {}
        if pca:
            self._set_pipe()
        self.model_only =  not pca

    def fit(self, X, y):
        self._gridsearch(X, y)
        self.features = X.columns

    def _set_pipe(self):
        self.params = {'est__' + key: val for key, val in self.params.items()}
        self.estimator = Pipeline([('pca', PCA()), ('est', self.estimator)])

    def _gridsearch(self, X, y, cv=LeaveOneOut(), n_jobs=-1):
        clf = GridSearchCV(estimator=self.estimator, param_grid=self.params, cv=cv,
                           scoring=self.metrics, refit='acc',
                           return_train_score=True, error_score=np.nan)
        clf.fit(X, y);
        self.gs_clf = clf
        self._format_results()

    def _format_results(self):
        results = pd.DataFrame(self.gs_clf.cv_results_)
        for param in self.params.keys():
            results[param] = results['params'].apply(lambda p: p[param])
        self.results = results

    def quick_results(self, df=False):
        param_list = [p for p, v in self.params.items() if len(v) > 1]
        res = self.results[param_list + ['mean_train_score', 'mean_test_score']]
        print(res)
        print()
        if self.model_only:
            print('est')
            print('----------')
            print(self.gs_clf.best_estimator_)
            print()
        else:
            for step_name, step in self.gs_clf.best_estimator_.steps:
                print(step_name)
                print('----------')
                print(step)
                print()
        if df:
            return res

    def get_best(self, all_steps=False):
        try:
            if all_steps or self.model_only:
                return self.gs_clf.best_estimator_
            else:
                return self.gs_clf.best_estimator_.steps[-1][1]
        except AttributeError:
            print('Model not fit...')


class Logit(Modeler):

    def __init__(self, params, metrics, pca):
        super().__init__(LogisticRegression(), params, metrics, pca)

    def feat_importances(self):
        if self.pca:
            if 'pca__n_components' in self.params:
                n_components = self.params['pca__n_components'][0]
                cols = ['pc_' + str(i) for i in range(1, n_components+1)]
            else:
                cols = ['pc_' + str(i) for i in range(1, len(self.features)+1)]
            coefs = np.squeeze(self.gs_clf.best_estimator_.steps[-1][1].coef_)
        else:
            cols = self.features
            coefs = np.squeeze(self.gs_clf.best_estimator_.coef_)
        return pd.Series(coefs, index=cols)


class SGDC(Modeler):

    def __init__(self, params, metrics):
        super().__init__(SGDClassifier(), params, metrics )

    def feat_importances(self):
        cols = self.features
        coefs = np.squeeze(self.gs_clf.best_estimator_.coef_)
        return pd.Series(coefs, index=cols)


if __name__ == '__main__':

    X, y = load_data()
    est = LogisticRegression()
    params = {'solver': ['saga'],
              'class_weight': ['balanced'],
              'penalty': ['l1'],
              'C': np.arange(0.5, 1.01, 0.1)}
    metrics = 'balanced_accuracy'

    model = Modeler(est, params, metrics, pca=False)
    model.fit(X, y)
    print('PCA: False')
    print(model.quick_results())

    model = Modeler(est, params, metrics, pca=True)
    model.fit(X, y)
    print('PCA: True')
    print(model.quick_results())
