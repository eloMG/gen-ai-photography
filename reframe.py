import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn



def Draw_thirds(height, width):
    plt.axhline(height / 3, color='blue')
    plt.axhline(2 * height / 3, color='blue')
    plt.axvline(width / 3, color='blue')
    plt.axvline(2 * width / 3, color='blue')

def shift_image(image_array, dx, dy, return_mask = False):
    new_image_array = np.full_like(image_array, -1)#negative value is for mask
    if dx > 0 and dy > 0:
        new_image_array[dy:, dx:,:] = image_array[0:-dy, 0:-dx,:] #shift image to the down and right
    elif dx > 0 and dy < 0:
        new_image_array[0:dy, dx:,:] = image_array[-dy:, 0:-dx,:] #shift image to the up and right
    elif dx < 0 and dy > 0:
        new_image_array[dy:, 0:dx,:] = image_array[0:-dy, -dx:,:] #shift image to the down and left
    elif dx < 0 and dy < 0:
        new_image_array[0:dy, 0:dx,:] = image_array[-dy:, -dx:] #shift image to the up and left
    
    mask = np.zeros((new_image_array.shape[0], new_image_array.shape[1]))
    mask[new_image_array[:,:,0] < 0] = 1
    
    #remove negative values
    new_image_array[new_image_array < 0] = 0
    
    if return_mask:
        return new_image_array, mask
    else:
        return new_image_array

def get_person_cordinate(image):
    """
    This function takes an image as input, segments it to identify the person in the image, 
    and extracts the average position of the person's head. The position is then returned 
    as a tuple of integers.

    Parameters:
    image (type): The input image. This should be in a format that the function can process 
                  to identify and segment the person.

    Returns:
    tuple: A tuple of two integers representing the average position of the person's head in the image.
    """
    
    #get the model
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # Define the class labels
    labels = {
        0: "Background",
        1: "Hat",
        2: "Hair",
        3: "Sunglasses",
        4: "Upper-clothes",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        8: "Belt",
        9: "Left-shoe",
        10: "Right-shoe",
        11: "Face",
        12: "Left-leg",
        13: "Right-leg",
        14: "Left-arm",
        15: "Right-arm",
        16: "Bag",
        17: "Scarf"
    }
    # List of class IDs you want to include
    class_ids = [11, 2, 1]  # Face, Hair, Hat class IDs to make head detection

    # Create a mask for the specific classes
    mask = torch.zeros_like(pred_seg)
    for class_id in class_ids:
        mask = mask | (pred_seg == class_id)

    # Apply the mask to the segmented image
    class_seg = mask.float()

    # Calculate the average position of the objects in the mask
    y_indices, x_indices = torch.where(mask)
    head_avg_position = [y_indices.float().mean().item(), x_indices.float().mean().item()]

    # Convert the position to integers
    head_avg_position = [int(round(pos)) for pos in head_avg_position]

    return head_avg_position



def refram_to_thirds(Image, Subject = None, Return_mask = False):
    
    
    return 0 # Placeholder
    