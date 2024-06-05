import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
from scipy.ndimage import zoom


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


def get_possible_subjects(image, confidence_threshold=0.92):
    #needs to test
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)


    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

    subjects = []
    for label, box in zip(results["labels"], results["boxes"]):
        subjects.append(model.config.id2label[label.item()])

    #only return unique subjects
    subjects = list(set(subjects))
    
    return subjects



def zoom_image_and_mask(image, mask, zoom_factor, origin):
    # Calculate the new shape of the image
    
    new_image = np.zeros_like(image)
    new_mask = np.ones_like(mask).astype(bool)
    
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

    # Zoom each color channel separately
    red_zoomed = zoom(red, zoom_factor, order=3)
    green_zoomed = zoom(green, zoom_factor, order=3)
    blue_zoomed = zoom(blue, zoom_factor, order=3)

    mask_zoomed = zoom(mask, zoom_factor, order=0)
    
    # Stack the zoomed color channels back into a single image
    zoomed_image = np.stack([red_zoomed, green_zoomed, blue_zoomed], axis=-1)
    
    
    
    new_origin = [int(origin[0] * zoom_factor), int(origin[1] * zoom_factor)]
    
    if zoom_factor > 1:
        shape = image.shape
        origin_shift = [new_origin[0] - origin[0], new_origin[1] - origin[1]]
        
        new_image[:,:,:] = zoomed_image[origin_shift[0]:shape[0]+origin_shift[0], origin_shift[1]:shape[1]+origin_shift[1], :]
        new_mask[:,:] = mask_zoomed[origin_shift[0]:shape[0]+origin_shift[0], origin_shift[1]:shape[1]+origin_shift[1]]
        
    else:
        shape = zoomed_image.shape
        origin_shift = [origin[0] - new_origin[0], origin[1] - new_origin[1]]
    
        
        new_image[origin_shift[0]:shape[0]+origin_shift[0], origin_shift[1]:shape[1]+origin_shift[1], :] = zoomed_image[:,:,:]
        new_mask[origin_shift[0]:shape[0]+origin_shift[0], origin_shift[1]:shape[1]+origin_shift[1]] = mask_zoomed[:,:]
    
    return new_image, new_mask


def refram_to_thirds(Image, Subject = None, Return_mask = False, show_focal_points = False, allow_zoom = True):
    width, height = Image.size
    
    #crop the image to get the subject and the cordinate boxes
    sub_images, boxes = crop_subjects(Image, "person", return_boxes = True)
    
    n_subjects = len(sub_images)
    
    #temp code to check the focal points will be removed in future
    print(f"Number of subjects: {n_subjects}")
    
    focal_points = []
    
    #some temporary variables
    zoom_factor = 1
    zoom_origin = (0,0)
    
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

    
    if show_focal_points:
        plt.imshow(Image)
        for focal_point in focal_points:
            plt.scatter(focal_point[1], focal_point[0], label = f"Focal point {focal_points.index(focal_point)}")
        plt.legend()
        plt.show()
    
    output = None
    
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
        
        if Return_mask:
            output_image, mask = shift_image(np.array(Image), dx, dy, return_mask = True)
        else:
            
            output_image = shift_image(np.array(Image), dx, dy, return_mask = Return_mask)
        
    else: 
        if n_subjects == 2:
            point_1 = focal_points[0]
            point_2 = focal_points[1]
        else:  # more than 2 subjects/focal points
            # Convert focal_points to a 2D numpy array
            focal_points = np.array(focal_points)

            # Compute the line vector (direction) from the first two focal points
            line_vector = focal_points[1] - focal_points[0]
            line_vector_norm = line_vector / np.linalg.norm(line_vector)

            # Project all focal points onto the line
            projections = np.dot(focal_points, line_vector_norm)

            # Find the indices of the two points with the minimum and maximum projections
            min_index = np.argmin(projections)
            max_index = np.argmax(projections)

            # These are the two extreme points
            point_1 = focal_points[min_index]
            point_2 = focal_points[max_index]
        
        point_1_norm = [point_1[0] / height, point_1[1] / width]
        point_2_norm = [point_2[0] / height, point_2[1] / width]
        
        #get line vector between the two points
        line_vector_norm = [point_2_norm[0] - point_1_norm[0], point_2_norm[1] - point_1_norm[1]]
        
        vector_ratio = line_vector_norm[0] / line_vector_norm[1]
        
        if abs(vector_ratio) > 2:
            #fit to vertical line
            
            if point_1_norm[0] > point_2_norm[0]:
                #flip values so that we enshure point 1 is on top
                point_1_norm, point_2_norm = point_2_norm, point_1_norm
            
            #calculate distance between the point in normelized spaced
            vertical_distance = abs(point_2_norm[0] - point_1_norm[0])
            
            zoom_factor = 0.5 / vertical_distance
            
            if point_1_norm[1] + point_2_norm[1] > 1:
                #Fit to the right line
                
                dx = (2/3) * width - (point_2[1] + point_1[1]) / 2
                dy = (1/2) * height - (point_2[0] + point_1[0]) / 2
                
                
                dx = int(round(dx))
                dy = int(round(dy))
                
                #temporary
                print(f"dx: {dx}, dy: {dy}")
                print("Vertical line, right")
                
                zoom_origin = (height // 2, 2* width // 3)
                
                
                
                
                
                
            else:
                #Fit to the left line
                
                
                dx = (1/3) * width - (point_2[1] + point_1[1]) / 2
                dy = (1/2) * height - (point_2[0] + point_1[0]) / 2
                
                
                dx = int(round(dx))
                dy = int(round(dy))
                
                
                #temporary
                print(f"dx: {dx}, dy: {dy}")
                print("Vertical line, left")
                
                
                zoom_origin = (height // 2, width // 3)
            

        elif abs(vector_ratio) > 0.5:
            #fit to diagonal line
            
            distance = np.sqrt((point_2_norm[0] - point_1_norm[0])**2 + (point_2_norm[1] - point_1_norm[1])**2)
            zoom_factor = 2**(1/2) / 2 / distance
            
            #temoprary
            print("Diagonal line")
            
            dx = (1/2) * width - (point_2[1] + point_1[1]) / 2
            dy = (1/2) * height - (point_2[0] + point_1[0]) / 2
            
            dx = int(round(dx))
            dy = int(round(dy))
            
            #temporary
            print(f"dx: {dx}, dy: {dy}")
            
            zoom_origin = (height // 2, width // 2)
            
        
        
        else:# abs(vector_ratio) < 0.5:
            #fit to horizontal line
            
            if point_1_norm[1] > point_2_norm[1]:
                #flip values so that we enshure point 1 is on left
                point_1_norm, point_2_norm = point_2_norm, point_1_norm
            
            horizontal_distance = abs(point_2_norm[1] - point_1_norm[1])
            
            zoom_factor = 0.5 / horizontal_distance
            
            
            if point_1_norm[0] + point_2_norm[0] > 1:
                #Fit to the bottom line
                
                dx = (1/2) * width - (point_2[1] + point_1[1]) / 2
                dy = (2/3) * height - (point_2[0] + point_1[0]) / 2
                
                
                dx = int(round(dx))
                dy = int(round(dy))
                
                #temporary
                print(f"dx: {dx}, dy: {dy}")
                print("Horizontal line, bottom")
                
                zoom_origin = (2 * height // 3, width // 2)
                
                
            else:
                #fit to the top line
                
                dx = (1/2) * width - (point_2[1] + point_1[1]) / 2
                dy = (1/3) * height - (point_2[0] + point_1[0]) / 2
                
                
                dx = int(round(dx))
                dy = int(round(dy))
                
                #temporary
                print(f"dx: {dx}, dy: {dy}")
                print("Horizontal line, top")
                
                zoom_origin = (height // 3, width // 2)
            
        #temoprary
        print(f"Line vector: {line_vector_norm}")
        
        
        #fit on line
        
        shifted_image, mask = shift_image(np.array(Image), dx, dy, return_mask = True)
        
        if allow_zoom:
            
            #Temporary
            print(f"zoom factor: {zoom_factor}, zoom origin: {zoom_origin}")
        
            output_image, mask = zoom_image_and_mask(shifted_image, mask, zoom_factor, zoom_origin)

        else:
            output_image = shifted_image
        
        
    if Return_mask:
        return output_image, mask
    else:
        return output_image

    