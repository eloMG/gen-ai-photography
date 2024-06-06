import numpy as np
from PIL import Image
import torch


from diffusers import StableDiffusionInpaintPipeline


def Infill(image, mask, prompt = ""):
    #checks if infill is needed
    infill_needed = mask.any()
    if infill_needed:
        init_image = Image.fromarray(image.astype('uint8'))
        mask_image = Image.fromarray(mask.astype('uint8')*255)

        #resize the images to 512x512 to fit the model
        init_image = init_image.resize((512, 512))
        mask_image = mask_image.resize((512, 512))


        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")

        infill_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        
        #resize the image back to the original size
        output_image = np.array(infill_image.resize((image.shape[1], image.shape[0])))
        
        #Replace part of image not infilled to keep original quality
        output_image[~mask] = image[~mask]
        
    else:
        print("No infill needed")
        output_image = image
    
    return output_image