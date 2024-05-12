import gdown
import os
import zipfile

def download_zip(url, save_path, name):
    """
    Download a zip file from a Google Drive URL and save it to a specified path
    """

    # Create the save path if it doesn't exist
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
    
    # Download scores (file not public, need to request access)
    download_zip(url='https://drive.google.com/uc?export=download&id=0BxeylfSgpk1MZ0hWWkoxb2hMU3c',
                 save_path='datasets/data',
                 name='imgListFiles_label')
    
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
    concatenated_content = '\n'.join(sorted(concatenated_content.split('\n')))
    # Remove empty lines from the concatenated content
    concatenated_content = '\n'.join(line for line in concatenated_content.split('\n') if line.strip())


    # Create the output file path
    output_file = os.path.join('datasets/data', 'aadb_scores.txt')

    # Write the concatenated content to the output file
    with open(output_file, 'w') as f:
        f.write(concatenated_content)
