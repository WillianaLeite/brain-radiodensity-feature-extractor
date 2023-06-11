import numpy as np
from numpy import ndarray
import pandas as pd
import yaml
from src.GaussianMixtureModel import GaussianMixtureModel
from logging import Logger
import os


class ExtractorGMM:

    """
    The instance extracts the features related to the brain regions of a hemisphere from the tomography.
    The features represent the segmentation of the brain in the regions defined in the configuration file.

    :param image: The input image.
    :type image: ndarray

    :param pixel_level_feature: If true, the segmentation returns the brain region of each pixel (MGABTD-pixel extractor).
                                If false returns the percentage of pixels in each brain region (MGABTD-percent).
    :type pixel_level_feature: bool

    :param logger: The main logging.
    :type logger: Logger
    """
    
    def __init__(self, image: ndarray, pixel_level_feature: bool, logger: Logger):

        self.image = image
        self.pixel_level_feature = pixel_level_feature
        self.logger = logger
        self.read_config()
        self.fit()

    def read_config(self):

        """ This function determines the value of the constants used to determine brain regions and to train the GMM. """

        try:
            with open(os.getcwd() + '/src/config.yml', 'r') as file:
                config = yaml.safe_load(file)

            self.N_ITER = int(config.get('N_ITER'))

            self.N_COMPONENTS_PER_CLASS = {}
            for c in config.get('N_COMPONENTS_PER_CLASS').split(', '):
                class_ = int(c.split('-')[0])
                n_componets = int(c.split('-')[1])
                self.N_COMPONENTS_PER_CLASS[class_] = n_componets

            self.NUMBER_REGIONS = len(self.N_COMPONENTS_PER_CLASS.keys())

            background_threshold = config.get('BACKGROUND').split('-')
            self.BACKGROUND = list(range(int(background_threshold[0]), int(background_threshold[1]) + 1))

            cerebrospinal_fluid_threshold = config.get('CEREBROSPINAL_FLUID').split('-')
            self.CEREBROSPINAL_FLUID = list(range(int(cerebrospinal_fluid_threshold[0]), int(cerebrospinal_fluid_threshold[1]) + 1))

            stroke_ischemic_threshold = config.get('STROKE_ISCHEMIC').split('-')
            self.STROKE_ISCHEMIC = list(range(int(stroke_ischemic_threshold[0]), int(stroke_ischemic_threshold[1]) + 1))

            white_matter_threshold = config.get('WHITE_MATTER').split('-')
            self.WHITE_MATTER = list(range(int(white_matter_threshold[0]), int(white_matter_threshold[1]) + 1))

            gray_matter_threshold = config.get('GRAY_MATTER').split('-')
            self.GRAY_MATTER = list(range(int(gray_matter_threshold[0]), int(gray_matter_threshold[1]) + 1))

            stroke_hemorrhagic_threshold = config.get('STROKE_HEMORRHAGIC').split('-')
            self.STROKE_HEMORRHAGIC = list(range(int(stroke_hemorrhagic_threshold[0]), int(stroke_hemorrhagic_threshold[1]) + 1))
            
            calcification_threshold = config.get('CALCIFICATION').split('-')
            self.CALCIFICATION = list(range(int(calcification_threshold[0]), int(calcification_threshold[1]) + 1))
            self.logger.info('Configuration variables read from configuration file')

        except:

            self.logger.warning('There is something wrong with the configuration file, read default values')
            self.N_COMPONENTS_PER_CLASS = {0:2, 1:2, 2:3, 3:2, 4:2, 5:3, 6:4}
            self.N_ITER = 5
            self.NUMBER_REGIONS = 7
            self.BACKGROUND = list(range(0, 2))
            self.CEREBROSPINAL_FLUID = list(range(0, 6))
            self.STROKE_ISCHEMIC = list(range(6, 22))
            self.WHITE_MATTER = list(range(23, 35))
            self.GRAY_MATTER = list(range(35, 41))
            self.STROKE_HEMORRHAGIC = list(range(50, 81))
            self.CALCIFICATION = list(range(130, 250))


    def fit(self):

        """ Training a GMM for classification using brain radiodensity regions as class. """

        list_rows = (
            list(zip([0] * len(self.BACKGROUND), self.BACKGROUND)) +
            list(zip([1] * len(self.CEREBROSPINAL_FLUID), self.CEREBROSPINAL_FLUID)) +
            list(zip([2] * len(self.STROKE_ISCHEMIC), self.STROKE_ISCHEMIC)) +
            list(zip([3] * len(self.WHITE_MATTER), self.WHITE_MATTER)) +
            list(zip([4] * len(self.GRAY_MATTER), self.GRAY_MATTER)) +
            list(zip([5] * len(self.STROKE_HEMORRHAGIC), self.STROKE_HEMORRHAGIC)) +
            list(zip([6] * len(self.CALCIFICATION), self.CALCIFICATION))
        )
        df_train = pd.DataFrame(list_rows, columns=['brain_region', 'hu_values'])
        df_train.drop_duplicates(inplace=True)
    
        self.gmm = GaussianMixtureModel(self.N_COMPONENTS_PER_CLASS, self.N_ITER)
        self.gmm.fit(df_train.drop(['brain_region'], axis=1), df_train['brain_region'])
        
    def segmentation(self):

        """ This function segments a brain tomography image into previously defined brain regions. """
        
        all_pixels = self.image.ravel()
        df = pd.DataFrame(all_pixels, columns=['hu_values'])

        # Each row is a pixel, and the columns are the classes.
        df_predict = self.gmm.predict(df)
        
        df_probability = df_predict.T
        segment = df_probability.idxmax()

        if self.pixel_level_feature:
            return segment
        else:
            classes = []
            for i in range(1, self.NUMBER_REGIONS + 1):
                classes.append(np.count_nonzero(segment == i))
            total = np.count_nonzero(segment != 0)
            probability = [count_classes/total for count_classes in classes]
            return probability