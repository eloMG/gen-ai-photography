import numpy as np
from PIL import Image
import torch


from diffusers import StableDiffusionInpaintPipeline


def Infill(image, mask, prompt = ""):

    image = image[:, :, :3] #remove alpha channel if present
    #checks if infill is needed
    infill_needed = mask.any()
    if infill_needed:
        init_image = Image.fromarray(image.astype('uint8'))
        mask_image = Image.fromarray(mask.astype('uint8') * 255)

        # Resize the images to 512x512 to fit the model
        init_image = init_image.resize((512, 512))
        mask_image = mask_image.resize((512, 512))

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=dtype
        )
        pipe = pipe.to(device)

        infill_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

        # Resize the image back to the original size
        output_image = np.array(infill_image.resize((image.shape[1], image.shape[0])))

        # Replace part of image not infilled to keep original quality
        output_image[~mask] = image[~mask]
        
    else:
        print("No infill needed")
        output_image = image
    
    return output_image