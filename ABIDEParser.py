# Part of the code in ABIDEParser.py is adopted from https://github.com/parisots/population-gcn

import os
import csv
import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
from scipy.spatial import distance
from utils import normalize, sparse_mx_to_torch_sparse_tensor

# Input data variables
root_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root_folder, "ABIDE_pcp/cpac/filt_noglobal")
phenotype = os.path.join(root_folder, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")

feature_names = ["SEX", "SITE_ID", "AGE_AT_SCAN"]
feat_length = len(feature_names)

# Get the list of subject IDs
def get_ids(num_subjects=None):
    """
    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, "subject_IDs.txt"), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["SUB_ID"] in subject_list:
                scores_dict[row["SUB_ID"]] = row[score]

    return scores_dict


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, atlas_name="aal", variable="connectivity"):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(
            data_folder, subject, subject + "_" + atlas_name + "_" + kind + ".mat"
        )
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    all_networks = np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [mat for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix


def get_labels():

    subject_IDs = get_ids()
    labels = get_subject_score(subject_IDs, score="DX_GROUP")
    labels = list(map(int, list(labels.values())))
    labels = np.array(labels) - 1
    return labels


def get_phenot_vector(subj_id):

    subject_IDs = get_ids()
    sites = get_subject_score(subject_IDs, score="SITE_ID")
    sites = np.array(np.unique(list(sites.values())))

    root_folder = os.path.dirname(os.path.abspath(__file__))
    phenotype = os.path.join(
        root_folder, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
    )

    df = pd.read_csv(phenotype)
    df.set_index("subject", inplace=True)
    features = df.loc[int(subj_id), feature_names].tolist()
    site_int_id = (np.where(sites == features[1]))[0][0]
    features[1] = site_int_id + 1
    return np.array(features)


def get_all_phenot_vectors():

    subject_IDs = get_ids()
    phenot_X = np.zeros((len(subject_IDs), feat_length))

    idx = 0
    for subj_id in subject_IDs:
        features = get_phenot_vector(subj_id)
        phenot_X[idx] = features
        idx += 1
    return phenot_X


def create_weighted_adjacency():

    phenot_X = get_all_phenot_vectors()
    Y = distance.pdist(phenot_X, "hamming") * 3
    Y = 3 - distance.squareform(Y)
    return Y


def get_num_edges():

    G = nx.from_numpy_matrix(create_weighted_adjacency())
    return G.number_of_edges()


def load_ABIDE(graph_type):

    atlas = "ho"
    connectivity = "correlation"

    # Get class labels
    subject_IDs = get_ids()
    labels = get_subject_score(subject_IDs, score="DX_GROUP")
    labels = np.array(list(map(int, list(labels.values())))) - 1
    num_nodes = len(subject_IDs)

    # Compute feature vectors (vectorised connectivity networks)
    features = get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Compute population graph using phenotypic features
    if graph_type == "original":
        final_graph = create_weighted_adjacency()
    if graph_type == "graph_no_features":
        final_graph = create_weighted_adjacency()
        features = np.identity(num_nodes)
    if graph_type == "graph_random":
        ones = get_num_edges() / (len(labels) * len(labels))
        final_graph = np.random.choice([0, 1], size=(len(labels), len(labels)), p=[1 - ones, ones])
        final_graph = (final_graph + final_graph.T) / 2
    if graph_type == "graph_identity":
        final_graph = np.zeros((num_nodes, num_nodes))

    final_graph = normalize(final_graph)

    adj = sp.coo_matrix(final_graph)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)

    features = sp.csr_matrix(features)
    features = normalize(features)

    # Convert to tensors
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)

    return adj, features, labels
