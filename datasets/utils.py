import gdown
import torch
import os
import zipfile

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


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
                 save_path='datasets/data',
                 name='datasetImages_originalSize')
    
    # Unzip scores
    if not os.path.exists('datasets/data/imgListFiles_label.zip'):
        # Unzip the contents
        with zipfile.ZipFile('datasets/data/imgListFiles_label.zip', 'r') as zip_ref:
            zip_ref.extractall('datasets/data/imgListFiles_label')
    
    # Get a list of all files in the directory
    label_path = 'datasets/data/imgListFiles_label'
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
    output_file = os.path.join('datasets/data', 'aadb_scores.txt')

    # Write the concatenated content to the output file
    with open(output_file, 'w') as f:
        f.write(concatenated_content)


class AADBDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path):

        save_aadb()
        transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
                ])
        self.dataset = ImageFolder(root=img_path, transform=transform)

        scores = []
        file_path = os.path.join(txt_path, 'aadb_scores.txt')
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