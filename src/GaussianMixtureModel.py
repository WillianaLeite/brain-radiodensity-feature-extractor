import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:

    def __init__(self, dict_k, iter=2):
        self.dict_k = dict_k 
        self.iter = iter

    def fit(self, X_train, y_train, verbose=False):

        self.df_train = X_train.copy()
        self.df_train['target'] = y_train
        self.list_class = sorted(list(set(y_train)))

        self.dict_priori = {}
        self.dict_gmm = {}
        self.dict_coef_mixture = {}
        for class_, k in self.dict_k.items():

            df_class = self.df_train[self.df_train['target'] == class_]
            coef_mixture = 1/k

            gm = GaussianMixture(
                n_components = k, 
                init_params = 'kmeans', 
                covariance_type = 'full', 
                n_init = self.iter, 
                random_state = 42
            ) 

            gm.fit(df_class.drop(['target'], axis=1).copy())
            self.dict_gmm[class_] = {}

            for component in range(k):
                self.dict_gmm[class_][component] = multivariate_normal(mean=gm.means_[component], 
                                                                       cov=gm.covariances_[component])
            
            self.dict_priori[class_] = len(df_class) / len(self.df_train)
            self.dict_coef_mixture[class_] = coef_mixture

    def predict(self, df):
        
        df_metrics = pd.DataFrame(df.reset_index(drop=True).index, columns=['index'])
        df_metrics.set_index('index')
        for class_ in sorted(self.list_class):
            for component in range(self.dict_k[class_]):
                df_metrics[f'class_{class_}_component_{component}'] = (
                    self.dict_gmm[class_][component].pdf(df) * self.dict_coef_mixture[class_]
                )

        for class_ in sorted(self.list_class):
            df_metrics[f'prob_gmm_class_{class_}'] = df_metrics[[f'class_{class_}_component_{component}' 
                                                                 for component in range(self.dict_k[class_])]].sum(axis=1)

            df_metrics[f'posteriori_{class_}'] = (
                df_metrics[f'prob_gmm_class_{class_}'] * 
                self.dict_priori[class_]
            )
        
        
        df_metrics['sum_posteriori'] = df_metrics[[f'posteriori_{class_}' 
                                                   for class_ in sorted(self.list_class)]].sum(axis=1)

        for class_ in sorted(self.list_class):
            df_metrics[class_] = df_metrics[f'posteriori_{class_}'] / df_metrics['sum_posteriori']

        return df_metrics[sorted(self.list_class)]