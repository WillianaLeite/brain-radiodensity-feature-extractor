import numpy as np
from numpy import ndarray
import pandas as pd
from src.GaussianMixtureModel import GaussianMixtureModel

N_COMPONENTS_PER_CLASS = {0:2, 1:2, 2:3, 3:2, 4:2, 5:3, 6:4}
BACKGROUND = list(range(0, 2))
CEREBROSPINAL_FLUID = list(range(0, 6))
STROKE_ISCHEMIC = list(range(6, 22))
WHITE_MATTER = list(range(23, 35))
GRAY_MATTER = list(range(35, 41))
STROKE_HEMORRHAGIC = list(range(50, 81))
CALCIFICATION = list(range(130, 250)) 

def _get_region_number(hu_value):
    if hu_value in BACKGROUND: 
        return 0
    elif hu_value in CEREBROSPINAL_FLUID: 
        return 1
    elif hu_value in STROKE_ISCHEMIC: 
        return 2
    elif hu_value in WHITE_MATTER: 
        return 3
    elif hu_value in GRAY_MATTER: 
        return 4
    elif hu_value in STROKE_HEMORRHAGIC: 
        return 5
    elif hu_value in CALCIFICATION: 
        return 6

class ExtractorGMM:
    
    def __init__(self, image: ndarray, pixel_level_feature: bool):

        '''
        Constructor that receives the image and chooses the output format of the features
        
        Parameters:
            image: The input image
            pixel_level_feature: When true, the features will be at the pixel level and when false, they will be percentage.
        '''

        self.image = image
        self.pixel_level_feature = pixel_level_feature
        self.fit()
    
    def fit(self):

        ''' Training a GMM for classification using brain radiodensity regions as class '''

        all_regions = BACKGROUND + CEREBROSPINAL_FLUID + STROKE_ISCHEMIC + WHITE_MATTER + GRAY_MATTER + STROKE_HEMORRHAGIC + CALCIFICATION

        hu_values = list(set(all_regions))
        df_train = pd.DataFrame(hu_values, columns=['hu_values'])
        df_train['brain_region'] = df_train['hu_values'].apply(_get_region_number)
    
        self.gmm = GaussianMixtureModel(N_COMPONENTS_PER_CLASS, 5)
        self.gmm.fit(df_train.drop(['brain_region'], axis=1), df_train['brain_region'])
        
    def segmentation(self):

        '''
        This function computes the features from a image
        
        Returns:
            Features from a input image
        '''
        
        all_pixels = self.image.ravel()
        df = pd.DataFrame(all_pixels, columns=['hu_values'])
        df_predict = self.gmm.predict(df) # Cada linha é um pixel, e as colunas são as classes
        
        df_probability = df_predict.T
        segment = df_probability.idxmax()

        if self.pixel_level_feature:
            return segment
        else:
            classes = []
            for i in range(1, 8): # São 7 classes
                classes.append(np.count_nonzero(segment == i))
            total = np.count_nonzero(segment != 0)
            probability = [count_classes/total for count_classes in classes]
            return probability