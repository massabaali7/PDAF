from transformers import AutoFeatureExtractor
def ast():
  feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593") 
  return feature_extractor