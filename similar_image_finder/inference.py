"""
Python script for finding similar images to a given image
fetched from the Firebase database storage.
"""

import csv
import torch
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
from torchvision import transforms
from data_loader.loader import Dataset, LoadData
from model.resnet_model import ResNetModel
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetModel(31)
print(os.getcwd())
model = model.load_state_dict('./model/backbone.pt', torch.device('cpu'))


def embeddings_loader(all_emb_path = 'data/all_embeddings.csv'):
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
    file = open(all_emb_path)
    dict_csv = csv.reader(file)
    #file.close()
    for row in dict_csv:
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
    embedding - Embedding of the image for which similar images are to be found.
    """
    image_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])

    img_transformed = image_transforms(img)
    img.close()

    # Transforming image to required format, and passing through the model.
    img_transformed = img_transformed.reshape(1, 3, 224, 224)
    embedding = model(img_transformed.to(device))

    # Saving the embedding as a 1D numpy array.
    embedding = embedding.cpu().detach().numpy()
    embedding = embedding.reshape(512,)
    return embedding


def cosine_calc(img_embedding, all_embeddings, file_paths):
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
        curr_emb = curr_emb.strip('][').split(', ')
        curr_emb = [float(x) for x in curr_emb]
        cos_sim = \
        dot(np.array(curr_emb), img_embedding)/(norm(np.array(curr_emb))*norm(img_embedding))
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
    for i in range(10):
        selected_img_paths.append(distances[i][1])

    return selected_img_paths


def similar_images_finder(img):
    """
    Find 10 similar images in order for the given image.
    ARGUMENTS
    img - Input image for which similar images are to be found out.

    Returns:
    similar_img_paths - image paths of 10 most similar images.
    """
    all_embeddings, all_file_paths = embeddings_loader('data/all_embeddings.csv')
    img_embedding = embedding_finder(img)
    similar_img_paths = cosine_calc(img_embedding, all_embeddings, all_file_paths)
    return similar_img_paths
