import numpy as np
import torch
import matplotlib.pyplot as plt

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



def refram_to_thirds(Image, Subject = None, Return_mask = False):
    
    return 0 # Placeholder
    