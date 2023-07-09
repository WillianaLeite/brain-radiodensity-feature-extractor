# Brain Feature Extractor
A simple library that extracts brain features from a Brain Tomography. The constructed features refer to the regions: 
- cerebrospinal fluid;
- calcification;
- gray matter;
- white matter;
- ischemic stroke;
- hemorrhagic stroke. 

> **_NOTE:_** The input image must be a file in DICOM format.

This extractor was proposed in Leite et al. (2022) [1] where it was used in the stroke classification process, obtaining 99.9% accuracy. In the article this extractor is called: Mixture Gaussian Analysis of Brain Tissue Density (MGABTD).

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
Link for more details: https://brain-radiodensity-feature-extractor.readthedocs.io/en/latest/source/api/BrainFeatureExtractorGMM.html

### References
1. W. L. S. Leite, R. M. Sarmento and C. M. J. M. Dourado Junior, "Feature extraction with mixture gaussian for stroke classification," 2022 35th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), Natal, Brazil, 2022, pp. 91-96, doi: 10.1109/SIBGRAPI55357.2022.9991801.
