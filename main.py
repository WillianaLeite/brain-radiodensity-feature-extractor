from src.BrainExtractorGMM import *

def main():

    x = BrainExtractorGMM(0.3, False)
    print(x.extract_features('sample/image157_isquemico.dcm', True))

main()