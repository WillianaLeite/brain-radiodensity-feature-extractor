from brain_feature_extractor import BrainFeatureExtractorGMM

class TestGaussianMixtureModel():

    def test_output_brain_extractor_GMM(self):
        extractor = BrainFeatureExtractorGMM(0.3, False)
        list_features = extractor.extract_features('sample/image157_isquemico.dcm', True)
        assert len(list_features) == 14