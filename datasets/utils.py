import gdown
import os
import zipfile

def download_aadb():
    """
    Dataset from paper https://arxiv.org/abs/1606.01621
    """

    url = 'https://drive.google.com/uc?id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_'
    save_path = '/datasets/data'

    # Create the save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    output = os.path.join(save_path, 'aadb.zip')

    gdown.download(url, output, quiet=False)

    # Unzip the contents
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    # Delete the zip file
    os.remove(output)