"""
Python script for finding similar images to a given image
fetched from the Firebase database storage.
"""

import csv
import torch
import tqdm
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from torchvision import transforms
from data_loader.loader import Dataset, LoadData
from model.resnet_model import ResNetModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetModel(31)
model = model.load_state_dict('backbone.pt')


def embeddings_loader(all_emb_path = 'all_embeddings.csv'):
    """
    Loading the RESNET embeddings for all images from
    all_embeddings.csv.
    ARGUMENTS:
    all_emb_path - File path for csv file which has all the
    embeddings stored.

    Returns:
    embeddings - Embeddings of all images in the dataset.
    file_paths - File paths for corresponding embeddings.
    """
    data = []
    with open(all_emb_path, 'r') as file:
        csvreader = csv.reader(file)
    for row in csvreader:
        data.append(row)

    file_paths = []
    embeddings = []
    for i in range(len(data)):
        file_paths.append(data[i][0])
        embeddings.append(data[i][1])
    return embeddings, file_paths

def embedding_finder(img):
    """
    Finding embedding of the input image.
    ARGUMENTS:
    img - Image for which embedding is to be found out.

    Returns:
    Embedding of the image for which similar images are to be found.
    """
    image_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])

    img_transformed = Dataset(pd.DataFrame(), "", image_transforms, img)
    load_data_train = LoadData(img_transformed, "", "test")
    data_loader = load_data_train.get_data_loader()

    with torch.no_grad():
        for input, _ in tqdm.tqdm(data_loader):
            input = input.to(device)
            output = model(input)
            embedding = output.cpu().detach().numpy()

    # Returning the embedding as a 1D numpy vector
    return embedding.flatten()


def similar_images_finder(img_embedding, all_embeddings, file_paths):
    """Find cosine similarity with all images in dataset
    and returning the 10 images with most similarity
    ARGUMENTS:
    img_embedding - Embedding of the image for which similar images are to be found.
    all_embeddings - Embeddings of all images in the dataset.
    file_paths - Corresponding file paths for images in all_embeddings.

    Returns:
    selected_img_paths - Image paths for similar images in order.
    """
    cos_dists = []

    for curr_emb in all_embeddings:
        cos_sim = \
        dot(curr_emb, img_embedding)/(norm(curr_emb)*norm(img_embedding))
        cos_dists.append(cos_sim)

    distances = []
    # Ensuring repeated images are not displayed multiple times.
    set_cos_dists = set()
    for i, cos_dist in enumerate(cos_dists):
        if cos_dist < 0.99 and cos_dist not in set_cos_dists:
            distances.append([cos_dist, file_paths[i]])
            set_cos_dists.add(cos_dist)

    distances = sorted(distances,key=lambda l:l[0], reverse=True)

    selected_img_paths = []
    for i in enumerate(10):
        selected_img_paths.append(distances[i][1])

    return selected_img_paths
