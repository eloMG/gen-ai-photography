from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

def get_segmentation_map(image, print_labels = False):

    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-ade")
    inputs = feature_extractor(images=image, return_tensors="pt")

    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-ade")
    outputs = model(**inputs)
   
    # you can pass them to feature_extractor for postprocessing
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    if print_labels:
        # Get the unique label values in the semantic segmentation map
        unique_labels = np.unique(predicted_semantic_map)
        class_labels = model.config.id2label
        # Print the class name for each unique label value
        for label in unique_labels:
            print(f"Label {label}: {class_labels[label]}")
        
    return predicted_semantic_map
    
from reframe import refram_to_thirds
def reframed_segmentation_map(image, subject = "person",print_labels = False):
    image_reframed = refram_to_thirds(image, Subject = subject)

    image_reframed = Image.fromarray(image_reframed.astype('uint8'))
    
    seg_map = get_segmentation_map(image_reframed, print_labels)
    
    return seg_map