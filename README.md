# Brain Feature Extractor
A simple library that extracts brain features from a Brain Tomography. The constructed features refer to the regions: 
- cerebrospinal fluid;
- clacification;
- gray matter;
- white matter;
- ischemic stroke;
- hemorrhagic stroke. 

> **_NOTE:_** The input image must be a file in DICOM format.

### Installation
```
pip install brain-radiodensity-feature-extractor
```

### Get started
How to extract features from a brain scan:

```Python
from brain_feature_extractor import BrainFeatureExtractorGMM

# Instancing the MGABTD-percent extractor
extractor = BrainFeatureExtractorGMM(percentage=0.3, pixel_level_feature=False)

# Extracting features from the image
print(extractor.extract_features(path='sample/image157_isquemico.dcm', verbose=True))
```
