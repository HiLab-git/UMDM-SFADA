import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy import ndimage

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def get_largest_component(image):
    dim = len(image.shape)
    if(image.sum() == 0 ):
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output

def test_single_volume(image, label, net, classes, patch_size=[40, 224, 224], deep_supervision=False):

    image = image.unsqueeze(0).cuda().float()
    label = label.squeeze(0).cpu().numpy() 
    prediction = np.zeros_like(label)
    net.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs, _ = net(image)
            if deep_supervision:
                outputs = outputs[0]
            else:
                outputs = outputs
            outputs = torch.softmax(outputs, dim=1) 
            predicted_classes = torch.argmax(outputs, dim=1)
        prediction = predicted_classes.squeeze(0).squeeze(0).cpu().numpy()
    metric_list = []
    for i in range(1, classes):
        metric = calculate_metric_percase(prediction == i, label == i)
        metric_list.append(metric)
    return metric_list

def test_single_volume_upl(image, label, net, classes, patch_size=[40, 224, 224], deep_supervision=False):

    image = image.unsqueeze(0).cuda().float()
    label = label.squeeze(0).cpu().numpy() 
    prediction = np.zeros_like(label)
    net.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = net(image)
            outputs = torch.softmax(outputs[0], dim=1) 
            predicted_classes = torch.argmax(outputs, dim=1)
        prediction = predicted_classes.squeeze(0).squeeze(0).cpu().numpy()
    metric_list = []
    for i in range(1, classes):
        metric = calculate_metric_percase(prediction == i, label == i)
        metric_list.append(metric)
    return metric_list
