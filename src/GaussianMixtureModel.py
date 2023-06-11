import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:

    """
    Pattern classifier that uses the Bayes theorem using the Mixture of Gaussians as a probability estimator.
    Each class has a Mixture of Gaussians and from the responsibility it is defined to which class the sample belongs.

    :param n_components_per_class: Number of Gaussian components each mixture of Gaussians will have in each class.
    :type n_components_per_class: dict

    :param n_iter: Maximum number of iterations that the Expectation Maximization algorithm will take to position the Gaussian components correctly.
    :type n_iter: int
    """

    def __init__(self, n_components_per_class: dict, n_iter: int = 2):
        self.n_components_per_class = n_components_per_class 
        self.n_iter = n_iter

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = False):

        """ Training the classifier from a set of samples with their respective classes. """

        self.df_train = X_train.copy()
        self.df_train['target'] = y_train
        self.list_class = sorted(list(set(y_train)))

        self.dict_priori = {}
        self.dict_gmm = {}
        self.dict_coef_mixture = {}
        for class_, k in self.n_components_per_class.items():

            df_class = self.df_train[self.df_train['target'] == class_]
            coef_mixture = 1/k

            gm = GaussianMixture(
                n_components = k, 
                init_params = 'kmeans', 
                covariance_type = 'full', 
                n_init = self.n_iter, 
                random_state = 42
            ) 

            gm.fit(df_class.drop(['target'], axis=1).copy())
            self.dict_gmm[class_] = {}

            for component in range(k):
                self.dict_gmm[class_][component] = multivariate_normal(mean=gm.means_[component], 
                                                                       cov=gm.covariances_[component])
            
            self.dict_priori[class_] = len(df_class) / len(self.df_train)
            self.dict_coef_mixture[class_] = coef_mixture

    def predict(self, df: pd.DataFrame):
        
        """ Given a set of input samples returns the class of each sample. """
        
        df_metrics = pd.DataFrame(df.reset_index(drop=True).index, columns=['index'])
        df_metrics.set_index('index')
        for class_ in sorted(self.list_class):
            for component in range(self.n_components_per_class[class_]):
                df_metrics[f'class_{class_}_component_{component}'] = (
                    self.dict_gmm[class_][component].pdf(df) * self.dict_coef_mixture[class_]
                )

        for class_ in sorted(self.list_class):
            df_metrics[f'prob_gmm_class_{class_}'] = df_metrics[[f'class_{class_}_component_{component}' 
                                                                 for component in range(self.n_components_per_class[class_])]].sum(axis=1)

            df_metrics[f'posteriori_{class_}'] = (
                df_metrics[f'prob_gmm_class_{class_}'] * 
                self.dict_priori[class_]
            )
        
        
        df_metrics['sum_posteriori'] = df_metrics[[f'posteriori_{class_}' 
                                                   for class_ in sorted(self.list_class)]].sum(axis=1)

        for class_ in sorted(self.list_class):
            df_metrics[class_] = df_metrics[f'posteriori_{class_}'] / df_metrics['sum_posteriori']

        return df_metrics[sorted(self.list_class)]