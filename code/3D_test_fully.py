import argparse
import os
import shutil
import random
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.unet import UNet, UNet_UPL
from dynamic_network.architectures.unet import PlainConvUNet, PlainConvUNet_UPL
# from networks.efficientunet import UNet
from scipy import ndimage
# from utils.distance_metrics_fast import hd95_fast, assd_fast
from utils.bezier_curve import bezier_curve, nonlinear_transformation
from utils.metrics import nsd
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='path/to/testing/dataset', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='exp/model/path', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--largest_component', type=bool, default=False,
                    help='get the largest component')
parser.add_argument('--deep_supervision', type=bool, default=False,
                    help='model enable deep supervision')
parser.add_argument('--nonlinear_aug', type=bool, default=False,
                    help='nonlinear transformation')
parser.add_argument('--confidence', type=float, default=None,
                    help='nonlinear transformation')

def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0 or gt.sum() == 0:
        return  0, float(np.nan), float(np.nan), 0  
    dice = metric.binary.dc(pred, gt)
    assd = metric.binary.assd(pred, gt, spacing)
    hd95 = metric.binary.hd95(pred, gt, spacing)
    ns = nsd(pred, gt, spacing, tolerance_mm=1)
    return dice, hd95, assd, ns

def get_largest_component(image):
    dim = len(image.shape)
    if (image.sum() == 0):
        return image
    if (dim == 2):
        s = ndimage.generate_binary_structure(2, 1)
    elif (dim == 3):
        s = ndimage.generate_binary_structure(3, 1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/{}.h5".format(case), 'r')
    image_org = h5f['image'][:]
    label = h5f['label'][:]
    org_spacing = h5f['spacing'][:]
    spacing = [org_spacing[2], org_spacing[1], org_spacing[0]]
    image = torch.from_numpy(image_org).unsqueeze(0).unsqueeze(0).cuda()
    prediction = np.zeros_like(label)
    if FLAGS.nonlinear_aug:
        image_repeated = image.repeat(8, 1, 1, 1, 1)
        transformed_images = image_repeated
        for i in range(8):
            transformed_images[i] = nonlinear_transformation(image_repeated[i])
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = net(transformed_images)
                outputs = outputs[0]
                outputs_p = torch.softmax(outputs, dim=1)
                outputs_e = outputs_p.mean(dim=0, keepdim=True)
                predicted_classes = torch.argmax(outputs_e, dim=1)
        if FLAGS.confidence is not None:
            outputs_e[outputs_e > FLAGS.confidence] = 1
            outputs_e[outputs_e < 1] = 0
            predicted_classes = torch.argmax(outputs_e, dim=1)
            prediction = predicted_classes.squeeze(0).squeeze(0).cpu().numpy()
        else:
            prediction = predicted_classes.squeeze(0).squeeze(0).cpu().numpy()
    else:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs= net(image)
                outputs = outputs[0]
                outputs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)

            prediction = predicted_classes.squeeze(0).squeeze(0).cpu().numpy()
    if FLAGS.largest_component:
        prediction = get_largest_component(prediction)
    first_metric = calculate_metric_percase(prediction == 1, label == 1, org_spacing)

    img_itk = sitk.GetImageFromArray(image_org.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric

def Inference(FLAGS):
    with open(FLAGS.root_path + '/testlist.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/".format(FLAGS.exp)
    pre = os.path.basename(FLAGS.root_path)
    test_save_path = "../model/{}/{}_{}_predictions/".format(
        FLAGS.exp, FLAGS.model, pre)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=FLAGS.num_classes,
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
    net.eval().cuda()

    dice_scores = []
    hd95_values = []
    assd_values = []
    nsd_values = []
    # Save metrics as CSV
    with open(os.path.join(test_save_path, 'metrics.csv'), 'w') as f:
        f.write("Case,Dice,HD95,ASSD,NSD\n")
        for case in tqdm(image_list):
            metrics = test_single_volume(case, net, test_save_path, FLAGS)
            dice_scores.append(metrics[0])
            hd95_values.append(metrics[1])
            assd_values.append(metrics[2])
            nsd_values.append(metrics[3])
            f.write(f"{case},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}\n")
    hd95_values = np.nan_to_num(hd95_values, nan=np.nanmax(hd95_values))
    assd_values = np.nan_to_num(assd_values, nan=np.nanmax(assd_values))
    # Calculate avg and std
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)

    avg_hd95 = np.mean(hd95_values)
    std_hd95 = np.std(hd95_values)

    avg_asd = np.mean(assd_values)
    std_asd = np.std(assd_values)

    avg_nsd = np.mean(nsd_values)
    std_nsd = np.std(nsd_values)

    avg_metrics = {
        'avg_dice': f"{avg_dice*100:.2f}±{std_dice*100:.2f}",
        'avg_hd95': f"{avg_hd95:.2f}±{std_hd95:.2f}",
        'avg_assd': f"{avg_asd:.2f}±{std_asd:.2f}",
        'avg_nsd': f"{avg_nsd*100:.2f}±{std_nsd*100:.2f}"
    }
    with open(os.path.join(test_save_path, 'overall_metrics.txt'), 'w') as f:
        for key, value in avg_metrics.items():
            f.write(f'{key}: {value}\n')

        return avg_metrics

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
        # print((metric[0]+metric[1]+metric[2])/3)