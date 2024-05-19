import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn



def Draw_thirds(height, width):
    plt.axhline(height / 3, color='blue')
    plt.axhline(2 * height / 3, color='blue')
    plt.axvline(width / 3, color='blue')
    plt.axvline(2 * width / 3, color='blue')

def shift_image(image_array, dx, dy, return_mask = False):
    new_image_array = np.zeros_like(image_array).astype(float)#negative value is for mask
    new_image_array.fill(-1)

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
    mask = mask.astype(bool)
    
    #remove negative values
    new_image_array[new_image_array < 0] = 0
    new_image_array = new_image_array.astype(np.uint8)
    
    if return_mask:
        return new_image_array, mask
    else:
        return new_image_array

def get_center_box(box):
    x1, y1, x2, y2 = box
    center = [(y1 + y2) / 2,  (x1 + x2)/ 2]
    return center


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

    #check if the mask is empty
    if class_seg.sum() == 0:
        #return center of the image
        return [image.size[1] // 2, image.size[0] // 2]
    
    # Calculate the average position of the objects in the mask
    y_indices, x_indices = torch.where(mask)
    head_avg_position = [y_indices.float().mean().item(), x_indices.float().mean().item()]

    # Convert the position to integers
    head_avg_position = [int(round(pos)) for pos in head_avg_position]

    return head_avg_position

def crop_subjects(image, subject, confidence_threshold=0.92, return_boxes=False):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)


    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

    subject = "person"#temp

    sub_images = []
    sub_boxes = []
    for label, box in zip(results["labels"], results["boxes"]):
        
        if model.config.id2label[label.item()] == subject:
            box = [round(i, 2) for i in box.tolist()]
            sub_image = image.crop(box)
            sub_images.append(sub_image)
            sub_boxes.append(box)

    if return_boxes:
        return sub_images, sub_boxes
    else:
        return sub_images



def refram_to_thirds(Image, Subject = None, Return_mask = False):
    width, height = Image.size
    
    #crop the image to get the subject and the cordinate boxes
    sub_images, boxes = crop_subjects(Image, "person", return_boxes = True)
    
    n_subjects = len(sub_images)
    
    #temp code to check the focal points will be removed in future
    print(f"Number of subjects: {n_subjects}")
    
    focal_points = []
    
    #check if we have any subjects
    if n_subjects == 0:
        raise ValueError(f"Subject: {Subject} not found in the image.")
    
    
    if Subject == "person":
        
        for sub_image, box in zip(sub_images, boxes):
            current_focal_point = get_person_cordinate(sub_image)
            #convert focal point to the image coordinate
            
            current_focal_point = [current_focal_point[0] + box[1], current_focal_point[1] + box[0]]
            
            focal_points.append(current_focal_point)
        
        
    else:
        for sub_image, box in zip(sub_images, boxes):
            current_focal_point = get_center_box(box)
            focal_points.append(current_focal_point)

    
    #temp code to check the focal points will be removed in future
    plt.imshow(Image)
    for focal_point in focal_points:
        plt.scatter(focal_point[1], focal_point[0], label = f"Focal point {focal_points.index(focal_point)}")
    plt.legend()
    plt.show()
    
    
    
    if n_subjects == 1:
        focal_point = focal_points[0]
        norm_head_avg_position = [focal_point[0] / height, focal_point[1] / width]
        norm_thirds = np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3], [1 / 3, 2 / 3], [2 / 3, 1 / 3]])

        norm_diff = [abs(norm_head_avg_position[0] - norm_third[0]) + abs(norm_head_avg_position[1] - norm_third[1]) for norm_third in norm_thirds]

        closesed_third_index = np.argmin(norm_diff)
        norm_diff_min = norm_diff[closesed_third_index]  # times six to get  as max value
        closesed_third = norm_thirds[closesed_third_index]

        #plt.scatter(closesed_third[1] * width, closesed_third[0] *height, color='pink')

        # Draw an arrow from the first point to the second point
        dx = closesed_third[1] * width - focal_point[1]
        dy = closesed_third[0] * height - focal_point[0]
        
        dx = int(round(dx))
        dy = int(round(dy))
        
    elif n_subjects == 2:
        point_1 = focal_points[0]
        point_2 = focal_points[1]
        
        point_1_norm = [point_1[0] / height, point_1[1] / width]
        point_2_norm = [point_2[0] / height, point_2[1] / width]
        
        #get line vector between the two points
        line_vector_norm = [point_2_norm[0] - point_1_norm[0], point_2_norm[1] - point_1_norm[1]]
        
        vector_ratio = line_vector_norm[0] / line_vector_norm[1]
        
        if abs(vector_ratio) > 2:
            #fit to vertical line
            
            #temoprary
            print("Vertical line")
            dx, dy = 0, 0
        elif abs(vector_ratio) > 0.5:
            #fit to diagonal line
            
            #temoprary
            print("Diagonal line")
            dx, dy = 0, 0
        else:# abs(vector_ratio) < 0.5:
            #fit to horizontal line
            
            #temoprary
            print("Horizontal line")
            dx, dy = 0, 0
        
        print(f"Line vector: {line_vector_norm}")
        
        
        #fit on line
        
        
        
        
        
        
    else:
        raise NotImplementedError("This function can only handle one subject for now.")  
    
    
    return shift_image(np.array(Image), dx, dy, return_mask = Return_mask)
    