from src.BrainExtractorGMM import BrainExtractorGMM

class TestGaussianMixtureModel():

    def test_output_brain_extractor_GMM(self):
        extractor = BrainExtractorGMM(0.3, False)
        list_features = extractor.extract_features('sample/image157_isquemico.dcm', True)
        assert len(list_features) == 14