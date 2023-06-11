from brain_feature_extractor import BrainFeatureExtractorGMM

# Instancing the MGABTD-percent extractor
extractor = BrainFeatureExtractorGMM(percentage=0.3, pixel_level_feature=False)

# Extracting features from the image
print(extractor.extract_features(path='sample/image157_isquemico.dcm', verbose=True))