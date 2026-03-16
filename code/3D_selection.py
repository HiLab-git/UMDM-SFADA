import argparse
import os
import random
import h5py
from tqdm import tqdm
import numpy as np

from typing import Dict, Callable, Optional, Tuple
from networks.unet import UNet
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from itertools import combinations 
from dataloaders.dataset_3d import threeDDataSet
import torch.nn as nn
from dynamic_network.architectures.unet import PlainConvUNet
from utils.tools import *
from utils.moments_tools import compute_spatial_moment_invariants, invariant_feature_distance
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/path/to/target/domain/dataset', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='path/to/source/model', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--tta_num', type=int, default=8,
                    help='num of test time augmentation')
parser.add_argument('--C', type=int, default=2, help='Hyperparameter: capacity of Du')
parser.add_argument('--mophology_enhancement', type=bool, default=True)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

def cluster_and_select_samples_for_combine_diversity(combined_data, k=3, initial_weight=0.01):

    features = [data[2] for data in combined_data]  # embeddings
    morphometric = [data[3] for data in combined_data]  # spatial moments
    X = np.array(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    embedding_dim = X_scaled.shape[1] 

    morpho_features = [list(morph_dict.values()) for morph_dict in morphometric]
    morpho_array = np.array(morpho_features)
    morpho_means = np.mean(morpho_array, axis=0)
    morpho_stds = np.std(morpho_array, axis=0) + 1e-10
    morpho_dim = morpho_array.shape[1] 
    
    def normalize_morph(morph_dict):
        values = [morph_dict[key] for key in sorted(morph_dict.keys())]
        return [(v - mean) / std for v, mean, std in zip(values, morpho_means, morpho_stds)]
    
    normalized_morphometric = [normalize_morph(m) for m in morphometric]
    
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_scaled)
    centroids = kmeans.cluster_centers_
    closest, distances_to_centroids = pairwise_distances_argmin_min(X_scaled, centroids)

    selected_indices = []
    for i in range(k):
        cluster_indices = np.where(closest == i)[0]
        local_min_index = np.argmin(distances_to_centroids[cluster_indices])
        selected_indices.append(cluster_indices[local_min_index])
    
    def objective_score(indices, weight):
        centroid_term = sum(distances_to_centroids[idx] for idx in indices) / embedding_dim
        
        morpho_selected = [normalized_morphometric[idx] for idx in indices]
        min_inter_cluster_distance = float('inf')

        for i in range(len(morpho_selected)):
            for j in range(i + 1, len(morpho_selected)):
                d = np.linalg.norm(np.array(morpho_selected[i]) - np.array(morpho_selected[j]))
                min_inter_cluster_distance = min(min_inter_cluster_distance, d)
                
        morpho_term = min_inter_cluster_distance / morpho_dim
        score = -centroid_term + weight * morpho_term

        print(f'[DEBUG] centroid: {centroid_term:.4f} | morpho: {morpho_term:.4f} | score: {score:.4f}')
        
        return score
    max_iterations = 10
    iteration = 0
    improved = True
    current_weight = initial_weight
    
    while improved and iteration < max_iterations:
        improved = False
        current_score = objective_score(selected_indices, current_weight)
        print(f'=== Iteration {iteration} === Weight: {current_weight:.6f} | Score: {current_score:.4f}')

        for cluster_idx in range(k):
            cluster_indices = np.where(closest == cluster_idx)[0]
            
            for candidate_idx in cluster_indices:
                if candidate_idx == selected_indices[cluster_idx]:
                    continue

                new_indices = selected_indices.copy()
                new_indices[cluster_idx] = candidate_idx

                new_score = objective_score(new_indices, current_weight)

                if new_score > current_score:
                    print(f'★ Improvement: Cluster {cluster_idx} | {selected_indices[cluster_idx]} -> {candidate_idx}')
                    print(f'★ Score: {current_score:.4f} -> {new_score:.4f}')
                    
                    selected_indices = new_indices
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                break
        
        iteration += 1

    selected_samples = [combined_data[idx] for idx in selected_indices]
    return selected_samples


def predict_with_tta_for_uncertainty_selection(image, net, output_path, parser, ratio=0.1):
    uncertainty = []
    feature_list = []
    img_name_list = []
    image_list = []
    pseudo_label = []
    label_list = []
    real_size = []
    size_esti = []
    morphometric = []
    m = parser.tta_num

    for case in tqdm(image):
        h5f = h5py.File(parser.root_path + "/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image_copy = image
        img_name = f'{case}.h5'
        img_name_list.append(img_name)
        params = []
        original_image = image_copy
        real_label = label
        pixel_count_real = np.count_nonzero(real_label)
        real_size.append(pixel_count_real)
        image_list.append(original_image)
        label_list.append(real_label)
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, m, axis=0)
        label = np.expand_dims(label, axis=0)
        label = np.repeat(label, m, axis=0)
        volume_batch = torch.from_numpy(image).float().unsqueeze(1)
        label_batch = torch.from_numpy(label).to(torch.int16).unsqueeze(1)
        for i in range(image.shape[0]):
            sample_data = {"image": volume_batch[i, :, :, :],
                            "label": label_batch[i, :, :, :]}
            transfrom = get_training_transforms(patch_size=np.array((128, 128, 128)), deep_supervision_scales=None,
                                                rotation_for_DA=None, mirror_axes=None, 
                                                do_dummy_2d_data_aug=False, do_spatial_transform=False)
            sample_data = transfrom(**sample_data)
            volume_batch[i, :, :, :] = sample_data["image"]
        original_view = torch.from_numpy(original_image).unsqueeze(
            0).unsqueeze(0).float().cpu()
        net.eval().cuda()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                volume_batch = volume_batch.cuda()
                original_view = original_view.cuda()
                out_put, _ = net(volume_batch)
                out_main, features = net(original_view)
                features = features[-1]
                features_np = features.squeeze().cpu().numpy()
                features = features_np.flatten()
                feature_list.append(features)
                out_put = torch.softmax(out_put, dim=1)
                out_put = torch.mean(out_put, dim=0, keepdim=True)
                pse_label = torch.argmax(out_put, dim=1).squeeze()
                pse_label = pse_label.cpu().numpy()
                pseudo_label.append(pse_label)
                entropy = softmax_entropy(out_put, softmax=False)
                entropy_np = entropy.cpu().detach().numpy()
                threshold = compute_entropy_density(entropy_np)
                selected_points = np.where(entropy_np > threshold)
                selected_mask = np.zeros_like(entropy_np, dtype=np.uint8)
                selected_mask[selected_points] = 1
                union_mask = np.logical_or(np.squeeze(selected_mask), pse_label).astype(np.uint8)
                spatial_moments = compute_spatial_moment_invariants(union_mask)
                GAUA_uncertainty = np.mean(entropy_np[selected_points])
                morphometric.append(spatial_moments)
                uncertainty.append(GAUA_uncertainty)
                # uncertainty.append(random.random())
                size_esti.append(len(selected_points[0]))

    budget = int(len(image_list) * ratio)
    print('The number of labeled slices is:',budget)
    
    combined_data = list(zip(img_name_list, uncertainty, feature_list, morphometric))
    combined_data.sort(key=lambda x: x[1], reverse=True)
    uncertainty_selected_samples = combined_data[:int(budget * parser.C)]
    # selected_samples = uncertainty_selected_samples
    method = f'UG_{parser.C}'
    if parser.size_regulization:
        selected_samples = cluster_and_select_samples_for_combine_diversity(uncertainty_selected_samples, k=budget)
        method = f'UMEM_{parser.C}'
    else:
        selected_samples = cluster_and_select_samples(uncertainty_selected_samples, k=budget)
    selected_img_names = [sample[0] for sample in selected_samples]

    all_data = list(zip(img_name_list, image_list, pseudo_label, label_list, uncertainty))
    for i, data in enumerate(all_data):
        img_name, image, pseudo_label, true_label, uncertainty = data
        if img_name in selected_img_names:
            all_data[i] = (img_name, image, true_label, -1)
        else:
            all_data[i] = (img_name, image, pseudo_label, uncertainty)
    all_data = [(img_name, image, pseudo_label, uncertainty) for img_name, image, pseudo_label, uncertainty in all_data]
    all_data.sort(key=lambda x: x[3])

    Dts_percent = int(len(all_data) * (1 - ratio * (parser.C - 1)))
    Dts_img_names = [data[0] for data in all_data[:Dts_percent]]
    with open(f"{output_path}/stage1_{method}.txt", "w") as f:
        for name in Dts_img_names:
            f.write(name + '\n')

    all_img_names = [data[0] for data in all_data]
    with open(f"{output_path}/all_{method}.txt", "w") as f:
        for name in all_img_names:
            f.write(name + '\n')

    with open(os.path.join(output_path, f'selection_{method}.txt'), 'w') as f:
        for item in selected_samples:
            f.write(f"{item[0]}: {item[1]}\n")


if __name__ == '__main__':
    parser = parser.parse_args()
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    torch.cuda.manual_seed(parser.seed)
    with open(parser.root_path + f'/trainlist.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/".format(parser.exp)
    net = PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=2,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}
    )
    save_mode_path = os.path.join(
        snapshot_path, 'UNet_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    predict_with_tta_for_uncertainty_selection(image_list, net, output_path=parser.root_path, parser=parser)
