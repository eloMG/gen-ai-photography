from utils import AADBDataset, get_prompts

dataset = AADBDataset('datasets/data/datasetImages_originalSize', 'datasets/data')

prompts = get_prompts(dataset)