import numpy as np
import pandas as pd
import cv2
import pydicom
import time
from brain_feature_extractor.ExtractorGMM import ExtractorGMM


class BrainFeatureExtractorGMM:

    """
    The instance extracts features related to brain regions from an entire brain scan.
    The features represent the segmentation of the brain in the regions defined in the configuration file.

    :param percentage: Input image reduction percentage.
    :type percentage: float

    :param pixel_level_feature: If true, the segmentation returns the brain region of each pixel (MGABTD-pixel extractor).
                                If false returns the percentage of pixels in each brain region (MGABTD-percent).
    :type pixel_level_feature: bool
    """

    def __init__(self, percentage: float = 0.3, pixel_level_feature: bool = False):

        self.percentage = percentage
        self.pixel_level_feature = pixel_level_feature

    def _load_dcm(self, img_path: str):

        """
        Loading a DICOM image of a brain scan.

        :param img_path: Local path with DICOM file.
        :type img_path: str

        :return: Array with the radiodensities of the DICOM image.
        :rtype: np.array
        """

        dcm = pydicom.read_file(img_path)
        rescale_intercept = int(dcm.data_element('RescaleIntercept').value)
        img = np.array(dcm.pixel_array, dtype=np.int16) + rescale_intercept
        return img
    
    def _extract_brain(self, src: np.array, inf_limit: int = 0, sup_limit: int = 100):

        """
        Removing the skull from the brain CT scan.

        :param src: Array with the radiodensities of the DICOM image.
        :type src: np.array

        :param sup_limit: Upper limit of Hounsfield Units (HU) that will be considered the skull region.
        :type sup_limit: int

        :param inf_limit: Lower limit of Hounsfield Units (HU) that will be considered the skull region.
        :type inf_limit: np.array

        :return: Array with the radiodensities of the DICOM image without the skull region.
        :rtype: np.array
        """

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

        """ Extract features of brain regions from a brain tomography DICOM file. """

        brain_image = self._load_dcm(path)
        new_img = self._extract_brain(brain_image, inf_limit=0, sup_limit=120)

        new_img = cv2.resize(new_img, (0, 0), fx=self.percentage, fy=self.percentage) 

        left_img = new_img[0:new_img.shape[0], 0:int(new_img.shape[1] / 2)]
        right_img = new_img[0:new_img.shape[0], int(new_img.shape[1] / 2):int(new_img.shape[1])]

        init_time = time.time()
        
        left_extractor = ExtractorGMM(left_img, self.pixel_level_feature)
        right_extractor = ExtractorGMM(right_img, self.pixel_level_feature)

        left_feat = left_extractor.segmentation()
        right_feat = right_extractor.segmentation()

        final_time = time.time() - init_time
        if verbose:
            print(f'Extract feature of {path} - TIME: {round(final_time, 2)}, seconds ...')

        features = left_feat + right_feat

        return [round(feat, 6) for feat in features]