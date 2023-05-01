import numpy as np
import pandas as pd
import cv2
import pydicom
import time
from collections import Counter
import multiprocessing
from GaussianMixtureModel import *


class ExtractorGMM:
    
    dict_GMM = {} # Variable of Class
    # Fazer um dicionario, cada configuracao vai ter um GMM diferente
    
    def __init__(self, image, k, classes, iter, percentage, pixel_level_feature):
        self.image = cv2.resize(image, (0, 0), fx=percentage, fy=percentage) 
        self.k = k
        self.n_classes = len(classes)
        self.classes = classes
        self.iter = iter
        self.config_gmm = f'classes: {str(classes)}; k: {str(k)}; iter: {iter}; percentage: {percentage}'
        self.all_pixels, self.all_classes = self.__find_roi()
        self.pixel_level_feature = pixel_level_feature
        self.manager = multiprocessing.Manager()
        self.probability = self.manager.list() 
        
    # Returns the class of the range the value belongs to
    def __get_class(self, dict_, val):
        for key, value in dict_.items():
            if val in value:
                return key
    
    def __get_neighborhood(self, point, nv=1):
        list_index = []
        qt_rows = self.rows-1
        qt_cols = self.cols-1
        for row in range(point[0] - nv, point[0] + nv + 1):
            for col in range(point[1] - nv, point[1] + nv + 1):
                if ((row >= point[0] - nv) and (row <= point[0] + nv) and 
                    (col >= point[1] - nv) and (col <= point[1] + nv) and
                    ((row, col) != point) and ((row, col) in self.dict_2d_1d.keys())):
                    list_index.append(self.dict_2d_1d[(row, col)])
        return list_index 
    
    # Find n points to be the lesion and background points
    def __find_roi(self):

        concatenated_pixels = self.image.ravel()

        all_classes = []
        list_hu_values = []
        for key in self.classes:
            inf_limit, sup_limit = self.classes[key].split('-')

            interval = range(int(inf_limit), int(sup_limit) + 1)
            self.classes[key] = list(interval)
            list_hu_values = list_hu_values + list(interval)
            all_classes.append(self.classes[key])

        self.df_train = pd.DataFrame(list_hu_values, columns=['hu_values'])
        self.df_train['target'] = self.df_train['hu_values'].apply(lambda value: self.__get_class(self.classes, value))
        
        # Map positions Array 2d to Array 1d
        self.dict_1d_2d = {}
        i = 0
        self.rows, self.cols = self.image.shape[:2]
        for row in range(self.rows):
            for col in range(self.cols):
                self.dict_1d_2d[i] = (row, col)
                i += 1
        self.dict_2d_1d = dict(zip(self.dict_1d_2d.values(), self.dict_1d_2d.keys()))
        
        return concatenated_pixels, all_classes
    
    def segmentation(self, window_radius): #window_radius=1 nível, 2 níveis, é sempre quadrado
        number_of_pixels = self.image.size
        
        if self.config_gmm in ExtractorGMM.dict_GMM.keys():
            gmm = ExtractorGMM.dict_GMM[self.config_gmm]
        else:
            ExtractorGMM.dict_GMM[self.config_gmm] = GaussianMixtureModel(self.k, self.iter, threshold=0.01)
            ExtractorGMM.dict_GMM[self.config_gmm].fit(self.df_train.drop(['target'], axis=1), self.df_train['target'])
            gmm = ExtractorGMM.dict_GMM[self.config_gmm]
        
        
        df = pd.DataFrame(self.all_pixels, columns=['hu_values'])
        df_predict = gmm.predict_proba(df) # Cada linha é um pixel, e as colunas são as classes
        
        if window_radius is None:
            df_probability = df_predict.T
            segment = df_probability.idxmax()
        else:
            df_class = pd.DataFrame(df_predict.T.idxmax(), columns=['class_GMM'])
            segment = []
            for pixel in range(number_of_pixels):
                row, col = self.dict_1d_2d[pixel]
                list_index = self.__get_neighborhood(point=(row, col), nv=window_radius)
                most_frequent_class = df_class.iloc[list_index]['class_GMM'].value_counts().index.tolist()[0]
                segment.append(most_frequent_class)
            segment = np.array(segment)
        if self.pixel_level_feature:
            self.probability.append(segment)
            return segment
        else:
            classes = []
            for i in range(1, self.n_classes):
                classes.append(np.count_nonzero(segment == i))
            total = np.count_nonzero(segment != 0)
            probability = [count_classes/total for count_classes in classes]
            self.probability.append(probability)
            return probability
        
class BrainExtractorGMM:
    
    def __init__(self, k, dict_class_extractor, percentage, iter, window_radius=None, 
                 enable_thread=False, pixel_level_feature=False):
        self.k = k
        self.iter = iter
        self.dict_class_extractor = dict_class_extractor
        self.percentage = percentage
        self.window_radius = window_radius
        self.enable_thread = enable_thread
        self.pixel_level_feature = pixel_level_feature
    
    def __load_dcm(self, img_path):
        dcm = pydicom.read_file(img_path)
        rescale_intercept = int(dcm.data_element('RescaleIntercept').value)
        img = np.array(dcm.pixel_array, dtype=np.int16) + rescale_intercept
        return img
    
    def __extract_brain(self, src, inf_limit=0, sup_limit=100):
      
        # Restrict the HU values to be between 0 and 255
        brain_image = np.where(src < inf_limit, 0, src)
        new_img = np.where(brain_image > sup_limit, 255, brain_image)
    
        # Get only the skull
        img = np.asarray(new_img, np.uint8)
        binary_image = np.where(img != 255, 0, img)
    
        # Remove the skull from the original image
        new_img = np.where(binary_image == 255, 0, new_img)
        new_img = np.where(new_img > sup_limit, 0, new_img)
    
        # Apply threshold
        ret, binary_image = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY)
        binary_image = np.asarray(binary_image, np.uint8)
    
        # Get the binaryImage biggest component
        connectivity = 4
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity, cv2.CV_8U)
    
        img_max = np.zeros(binary_image.shape, binary_image.dtype)
        large_component = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        img_max[labels == large_component] = 255
        img_max[labels != large_component] = 0
    
        # Compare the biggest component with the original image and get only the intersection
        output = np.where(img_max == 255, src, 0)
    
        return output

    def extract_features(self, path, verbose=False):
        brain_image = self.__load_dcm(path)
        new_img = self.__extract_brain(brain_image, inf_limit=0, sup_limit=120)

        left_img = new_img[0:new_img.shape[0], 0:int(new_img.shape[1] / 2)]
        right_img = new_img[0:new_img.shape[0], int(new_img.shape[1] / 2):int(new_img.shape[1])]

        init_time = time.time()
        
        left_extractor = ExtractorGMM(image=left_img, 
                                      k=self.k, 
                                      classes=self.dict_class_extractor.copy(), 
                                      percentage=self.percentage,
                                      iter=self.iter,
                                      pixel_level_feature=self.pixel_level_feature)
        right_extractor = ExtractorGMM(image=right_img, 
                                       k=self.k, 
                                       classes=self.dict_class_extractor.copy(), 
                                       percentage=self.percentage,
                                       iter=self.iter,
                                       pixel_level_feature=self.pixel_level_feature)
        
        if self.enable_thread:
            
            # More information: https://stackoverflow.com/questions/26239695/parallel-execution-of-class-methods
            thead_left = multiprocessing.Process(target=left_extractor.segmentation, args=(self.window_radius, ))
            thread_right = multiprocessing.Process(target=right_extractor.segmentation, args=(self.window_radius, ))
            thead_left.start()
            thread_right.start()
            thead_left.join()
            thread_right.join()
            left_proba = left_extractor.probability[0]
            right_proba = right_extractor.probability[0]
        else:
            left_proba = left_extractor.segmentation(self.window_radius)
            right_proba = right_extractor.segmentation(self.window_radius)
                                    
        final_time = time.time() - init_time
        if verbose:
            print(f'Extract feature of {path} - TIME: {round(final_time, 2)}, seconds ...')
        
        probability = left_proba + right_proba
        return [float('{:.6f}'.format(x)) for x in probability], final_time