import gdown
import torch
import os
import zipfile

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import BitsAndBytesConfig, pipeline
from tqdm import tqdm


def download_zip(url, save_path, name):
    """
    Download a zip file from a Google Drive URL and save it to a specified path
    """

    # Create the save path if it doesn't exist
    save_path = os.path.join(save_path, name)
    os.makedirs(save_path, exist_ok=True)

    output = os.path.join(save_path, f'{name}.zip')

    if not os.path.exists(os.path.join(save_path, f'{name}')):
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        # Unzip the contents
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(save_path)


def save_aadb():
    """
    Dataset from paper https://arxiv.org/abs/1606.01621
    """
    
    # Download images
    download_zip(url='https://drive.google.com/uc?id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_',
                 save_path='dataset/data',
                 name='datasetImages_originalSize')
    
    # Unzip scores
    if not os.path.exists('dataset/data/imgListFiles_label'):
        # Unzip the contents
        with zipfile.ZipFile('dataset/data/imgListFiles_label.zip', 'r') as zip_ref:
            zip_ref.extractall('dataset/data/imgListFiles_label')
    
    # Get a list of all files in the directory
    label_path = os.path.join('dataset', 'data', 'imgListFiles_label')
    file_list = os.listdir(label_path)

    # Filter the list to include only files ending with "_score"
    score_files = [file for file in file_list if file.endswith("_score.txt")]

    # Create an empty string to store the concatenated content
    concatenated_content = ""

    # Iterate over each score file
    for file in score_files:
        # Open the file and read its content
        with open(os.path.join(label_path, file), 'r') as f:
            content = f.read()
            
            # Concatenate the content to the existing string
            concatenated_content += content

    # Rearrange the lines in alphabetical order
    concatenated_content = '\n'.join(sorted(set(concatenated_content.split('\n'))))
    # Remove empty lines from the concatenated content
    concatenated_content = '\n'.join(line for line in concatenated_content.split('\n') if line.strip())


    # Create the output file path
    output_file = os.path.normpath(os.path.join('dataset/data', 'aadb_scores.txt'))

    # Write the concatenated content to the output file
    with open(output_file, 'w') as f:
        f.write(concatenated_content)


class AADBDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path):

        if not os.path.exists(os.path.normpath(os.path.join(txt_path, 'aadb_scores.txt'))):
            save_aadb()

        transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
                ])
        self.dataset = ImageFolder(root=img_path, transform=transform)

        scores = []
        file_path = os.path.normpath(os.path.join(txt_path, 'aadb_scores.txt'))
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.split()
                scores.append(float(columns[1]))

        self.scores = torch.tensor(scores)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {'image': self.dataset[idx][0], 'score': self.scores[idx]}
        return sample
    

def get_prompts(dataset):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "llava-hf/llava-1.5-7b-hf"

    pipe = pipeline("image-to-text",
                    model=model_id,
                    model_kwargs={"quantization_config": quantization_config})

    prompt = "USER: <image>\nProvide a short description of the image, to use as prompt for an image generating model\nASSISTANT:"

    prompts = []
    for i in tqdm(range(len(dataset) // 3)):
        image = transforms.functional.to_pil_image(dataset[i]['image'])
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        prompts.append(outputs[0]['generated_text'][108:])
    
    with open('dataset/data/aadb_prompts.txt', 'w') as file:
        for string in prompts:
            file.write(string + '\n')

    return prompts