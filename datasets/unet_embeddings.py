import torch
import numpy as np

from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from utils import AADBDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

dataset = AADBDataset('datasets/data/datasetImages_originalSize', 'datasets/data')

with open('datasets/data/aadb_prompts.txt', 'r') as file:
    lines = file.readlines()
    
prompts = [line.strip() for line in lines]

latents = []

for i in tqdm(range(len(prompts))):
    prompt = prompts[i]
    image = dataset[i]['image']
    image = image.to(device=device, dtype=torch.half).unsqueeze(0)
    latent = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5, output_type='latent').images
    latents.append(latent.detach().cpu().numpy())

latents = np.array(latents).squeeze()

np.save('datasets/data/saved/unet_latents.npy', latents)