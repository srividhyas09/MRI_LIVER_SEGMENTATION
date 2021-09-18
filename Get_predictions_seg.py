from prediction_preprocess import *
import torch
from unet import UNet
from dual_tail_net import Dual
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import morphology


def cmpt_contour_metrics(gt, pred):
    sampling = (1, 1)
    connectivity = 1
    gt = np.atleast_1d(gt.astype(np.bool))
    pred = np.atleast_1d(pred.astype(np.bool))
    # Generate binary filter structure to erode the u_net masks for one pixel border
    conn = morphology.generate_binary_structure(gt.ndim, connectivity)
    # Find contours on both the gt u_net maks as well as the prediction u_net mask
    # Exploit logical XOR operation to only get the previously eroded "1"-pixel border
    contours_gt = np.logical_xor(gt, morphology.binary_erosion(gt, conn))
    contours_pred = np.logical_xor(pred, morphology.binary_erosion(pred, conn))
    # We have to keep track of the contour pixels as to be able to later compute a reliable mean.
    # Without adding zeros, perfectly segmented pixels won't be accounted in the calculation of ASD metric.
    num_contourpx_gt = np.count_nonzero(contours_gt)
    num_contourpx_pred = np.count_nonzero(contours_pred)
    # Build euclidean distance maps for the ground truth u_net map as well as for the predictions.
    # Invert the contours_map as we want to have the surface pixels to have a weight of zero.
    distance_map_gt = morphology.distance_transform_edt(~contours_gt, sampling)
    distance_map_pred = morphology.distance_transform_edt(~contours_pred, sampling)
    # Multiply ground truth distance map onto the contours pixels of the predicted u_net mask
    # Also possible via: np.ravel(distance_map_gt[contours_pred != 0])
    pred2gt_distance_map = distance_map_gt * contours_pred
    # Multiply prediction distance map onto the contours pixels of the ground truth u_net mask
    # Also possible via: np.ravel(distance_map_pred[contours_gt != 0])
    gt2pred_distance_map = distance_map_pred * contours_gt
    pred2gt_distances = list(pred2gt_distance_map[pred2gt_distance_map != 0])
    pred2gt_distances += list(np.zeros(num_contourpx_pred - len(pred2gt_distances)))
    gt2pred_distances = list(gt2pred_distance_map[gt2pred_distance_map != 0])
    gt2pred_distances += list(np.zeros(num_contourpx_gt - len(gt2pred_distances)))
    symmetric_distances = np.concatenate([pred2gt_distances, gt2pred_distances])
    valid = len(symmetric_distances) > 0 and np.max(gt) > 0
    surface_metrics = {
        "mean_sd"  : np.mean(symmetric_distances) if valid else np.nan,
        "median_sd": np.median(symmetric_distances) if valid else np.nan,
        "std_sd"   : np.std(symmetric_distances) if valid else np.nan,
        "max_sd"   : np.max(symmetric_distances) if valid else np.nan,
        "rmsd"     : np.sqrt(1 / len(symmetric_distances)) * np.sqrt(
            np.sum(np.square(symmetric_distances))) if valid else np.nan,
    }
    return surface_metrics


if __name__ == '__main__':
    # load sample image
    data_transform_augment = transforms.Compose([ResizePadCrop(), ToTensor()])
#    data_transform_augment = transforms.Compose([ResizePadCrop(), Normalization(), ToTensor()])

    #dataset = Preprocessing('../thesis/ground_truth/NASH_grading_1.csv', transform=data_transform_augment)
    dataset = Preprocessing('ground_truth/test.csv', transform=data_transform_augment)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_path = 'saved_models/Dual_train_ZeroToOne.pt'
    #model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = Dual(n_channels=1, n_classes=1, bilinear=True)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    criterion = nn.MSELoss()
    angle_error = []
    count = 0
    for batch_index, sample in enumerate(dataloader):
        count += 1
        inputs = sample['image']

        image = inputs.type(torch.FloatTensor)

        preds = model.forward(image)

        actual = sample['mask']
        new = np.array(preds.squeeze().detach())

        #new[new < 0.5] = 0
        #new[new >= 0.5] = 255

        # actual[actual != 63] = 0
        # fin = cmpt_contour_metrics(np.array(actual.squeeze().detach()), new)
        # print(fin)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(actual.squeeze())  # row=0, col=0
        ax[1].imshow(new)  # row=0, col=1
        plt.title(count)
        plt.show()
