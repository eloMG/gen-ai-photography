from utils import AADBDataset, get_prompts

dataset = AADBDataset('dataset/data/datasetImages_originalSize', 'dataset/data')

prompts = get_prompts(dataset)