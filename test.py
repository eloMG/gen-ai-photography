from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

def Draw_thirds(height, width):
    plt.axhline(height / 3, color='blue')
    plt.axhline(2 * height / 3, color='blue')
    plt.axvline(width / 3, color='blue')
    plt.axvline(2 * width / 3, color='blue')

def shift_image(image_array, dx_int, dy_int, return_mask = False):
    new_image_array = np.full_like(image_array, -1)#negative value is for mask
    if dx_int > 0 and dy_int > 0:
        new_image_array[dy_int:, dx_int:,:] = image_array[0:-dy_int, 0:-dx_int,:] #shift image to the down and right
    elif dx_int > 0 and dy_int < 0:
        new_image_array[0:dy_int, dx_int:,:] = image_array[-dy_int:, 0:-dx_int,:] #shift image to the up and right
    elif dx_int < 0 and dy_int > 0:
        new_image_array[dy_int:, 0:dx_int,:] = image_array[0:-dy_int, -dx_int:,:] #shift image to the down and left
    elif dx_int < 0 and dy_int < 0:
        new_image_array[0:dy_int, 0:dx_int,:] = image_array[-dy_int:, -dx_int:] #shift image to the up and left
    
    mask = np.zeros((new_image_array.shape[0], new_image_array.shape[1]))
    mask[new_image_array[:,:,0] < 0] = 1
    
    #remove negative values
    new_image_array[new_image_array < 0] = 0
    
    if return_mask:
        return new_image_array, mask
    else:
        return new_image_array

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

img_path = "Test_images/24500d84-4f62-46fc-991e-854e292b18c8.webp"

image = Image.open(img_path)

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
class_ids = [11, 2, 1]  # Replace with your class IDs

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

# Display the image with the average position marked
#plt.imshow(class_seg)
plt.imshow(image)
plt.imshow(pred_seg, cmap = "hsv", alpha = 0.3)

plt.scatter(head_avg_position[1], head_avg_position[0], color='red')  # Note that the x and y positions are swapped for plotting

# Get the size of the image
height, width = class_seg.shape

# Draw lines that divide the image into three equal parts in the x and y directions
Draw_thirds(height, width)

#Finding normelized difference between center of head and thirds of image
norm_head_avg_position = [head_avg_position[0] / height, head_avg_position[1] / width]
norm_thirds = np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3], [1 / 3, 2 / 3], [2 / 3, 1 / 3]])
norm_diff = [abs(norm_head_avg_position[0] - norm_third[0]) + abs(norm_head_avg_position[1] - norm_third[1]) for norm_third in norm_thirds]

closesed_third_index = np.argmin(norm_diff)
norm_diff_min = norm_diff[closesed_third_index]  # times six to get  as max value
closesed_third = norm_thirds[closesed_third_index]

# Draw an arrow from the first point to the second point
dx = closesed_third[1] * width - head_avg_position[1]
dy = closesed_third[0] * height - head_avg_position[0]
plt.arrow(head_avg_position[1], head_avg_position[0], dx, dy, color='pink')

plt.show()
#plt.close()

#Shifting image to thirds

dx_int = int(round(dx))
dy_int = int(round(dy))

img_arr = np.array(image)
shifted_image_array = shift_image(img_arr, dx_int, dy_int)

Draw_thirds(height, width)
plt.imshow(shifted_image_array)
plt.show()

