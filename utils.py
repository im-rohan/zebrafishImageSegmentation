import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import tifffile
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from patchify import patchify, unpatchify 
#from torchvision import transform

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
 
        return loss / self.n_classes

'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=100):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.reduction=reduction
        self.ignore_index = ignore_index
    def forward(self, inputs, targets):
        p=F.softmax(imputs, dim=1)
        #get log probablities
        ce_loss=F.cross_entropy(input, targets, reduction='none', ignore_index=self.ignore_index)

        # Get probablitites for true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        #Caclculate focal loss
        focal_weight = (1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = focal_weight*ce_loss

        #Apply alpha weighting if specifies
        if self.alpha is not None:
            if insinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                #Alpha is a list/tensor of weights per class
                alpha = torch.tensor(self, device=inputs.device, dype=inputs.dtype)
                alpha_t = alpha.gather(0, targets.flatten()).view_as(targets)
            focal_loss=alpha_t*focal_loss
        if self.reduction=='mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
'''    


# class FocalLoss(nn.Module):
#     def __init__(self,  gamma=2, alpha= None, reduction='mean', num_classes=None):
#         '''
#         Unified Focal Loss class for multiclass calsffiction
#         :param gamma: Focusing parameter, controls the strength of moduating factor (1 - p_t)^gamma
#         :param alpha: Balancing factor, can be scalar or a tensor for class-wise weights. If None, no class blanacing is used.
#         :param reductiom: SPecifies the reduction method:'none'|'mean'|'sum'
#         :param task_type: Specifies the type of task:'binary','multi-class' or'multi-label'
#         :param num_classes: Number of clases (only requred for multi-class classification)
#         '''
#         super(FocalLoss, self).__init__()
#         self.gamma=gamma
#         self.alpha=alpha
#         self.reduction=reduction
#         self.task_type=task_type
#         self.num_classes=num_classes

#         #Handle alpha for class balancing in multiclass task
        
#         if alpha is not None and isinstance(alpha, (list, torch.Tensor)):
#             assert num_classes is not None, "num_classes must be specified for multiclass classificaiton"
        

#     def forward(self, inputs, targets):
#         '''Focal loss for multiclass-classfiction.'''
#         if self.alpha is not None:
#             alpha=self.alpha.to(inputs.device)

#         #Convert logits to probablitites with softmax
#         probs = F.softmax(inputs, dim=1)

#         # One-hot encode the targets
#         targets_one_hot=F.one_hot(targets, num_classes=self.num_classes).float()
        
#         #Compute cross entropy loss for each class
#         ce_loss = -targets_one_hot * torch.log(probs)

#         p_t = torch.sum(probs*targets_pne_hot, dim=1)
#         focal_weight = (1-p_t)**self.gamma

#         if self.alpha is not None:
#             alpha_t = 

        
        
def load_files(data_path):
    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} does not exist" )
    if os.path.exists(data_path):
        print(len(os.listdir(data_path)))
        file_list=[]
        for file in os.listdir(data_path):
            # print(file)
            file_path = os.path.join(data_path, file)
            # print(file_path)
            file_list.append(file_path)

        return file_list
   

def extract_annotated_image_test(image, multi_label_mask):
    annotated_slices=[]
    image_np = np.array(image)

    mask_np=np.array(multi_label_mask)
    image_np_shape=image_np.shape
    mask_np_shape=mask_np.shape
    #print(f"The shape of the image is {image_np_shape}")
    #print(f"The shape of the image is {mask_np_shape}")
    
    if mask_np.shape[0] > image_np.shape[0]:

        z_difference= mask_np.shape[0]-image_np.shape[0]

        top_slices= z_difference//2

        bottom_slices=z_difference-top_slices

        mask_np= mask_np[top_slices:-bottom_slices,:,:]

    elif(mask_np.shape[0] < image_np.shape[0]):

        z_difference= image_np.shape[0]-mask_np.shape[0]

        top_slices= z_difference//2

        bottom_slices=z_difference-top_slices
   
        image_np= image_np[top_slices:-bottom_slices,:,:]
   
    for i in range(mask_np.shape[0]):

        if np.sum(mask_np[i,:,:])>600000:

            annotated_slices.append(i)
    #print(f"The annotated slices for the image is {annotated_slices}")
    image_np = image_np[annotated_slices,:,:]

    mask_np = mask_np[annotated_slices,:,:]
    
    return image_np, mask_np
    
def extract_annotated_image_train(train_image_list, train_mask_list, directory_path_image = "/projects/wan-lab/zebrafish/annotations_harshil/train_images_2d", directory_path_label = "/projects/wan-lab/zebrafish/annotations_harshil/train_labels_2d"):

    if len(train_image_list)!=len(train_mask_list):
        raise ValueError("The length of the two list should be equal")
    
    for image_name, label_name in zip(train_image_list, train_mask_list):
        print(image_name)
        print(label_name)
        annotated_slices=[]
        image=tifffile.imread(image_name)
        label=tifffile.imread(label_name)
        image_np = np.array(image)
        label_np=np.array(label)
        image_np_shape=image_np.shape
        print(image_np_shape)
        label_np_shape=label_np.shape
        print(label_np_shape)
        if label_np.shape[0] > image_np.shape[0]:
    
            z_difference= label_np.shape[0]-image_np.shape[0]
    
            top_slices= z_difference//2
    
            bottom_slices=z_difference-top_slices
    
            label_np= label_np[top_slices:-bottom_slices,:,:]
    
        elif(label_np.shape[0] < image_np.shape[0]):
    
            z_difference= image_np.shape[0]-label_np.shape[0]
    
            top_slices= z_difference//2
    
            bottom_slices=z_difference-top_slices
       
            image_np= image_np[top_slices:-bottom_slices,:,:]
       
        for i in range(label_np.shape[0]):
    
            if np.sum(label_np[i,:,:])>600000:
    
                annotated_slices.append(i)
    
        image_np = image_np[annotated_slices,:,:]
    
        label_np = label_np[annotated_slices,:,:]
    
        for i in range((image_np.shape[0])):
            image_slice=image_np[i]
            
            label_slice=label_np[i]
    
            os.makedirs(directory_path_image, exist_ok=True)
            os.makedirs(directory_path_label, exist_ok=True)
            image_name_slice = os.path.basename(image_name[:-4])+f"slice_{i}.tif"
            image_path = os.path.join(directory_path_image, image_name_slice)
            label_name_slice = os.path.basename(label_name[:-4])+f"slice_{i}.tif"
            label_path = os.path.join(directory_path_label, label_name_slice)
            tifffile.imwrite(image_path,image_slice)
            tifffile.imwrite(label_path, label_slice)
    print("All_image and mask_slices saved")

def extract_annotated_image_train_augment(train_image_list, train_mask_list, transform, directory_path_image = "/projects/wan-lab/zebrafish/annotations_harshil/train_images_2d_augmented", directory_path_label = "/projects/wan-lab/zebrafish/annotations_harshil/train_labels_2d_augmented"):

    if len(train_image_list)!=len(train_mask_list):
        raise ValueError("The length of the two list should be equal")
    
    for image_name, label_name in zip(train_image_list, train_mask_list):
        #print(image_name)
        #print(label_name)
        annotated_slices=[]
        image=tifffile.imread(image_name)
        label=tifffile.imread(label_name)
        image_np = np.array(image)
        label_np=np.array(label)
        image_np_shape=image_np.shape
        print(image_np_shape)
        label_np_shape=label_np.shape
        print(label_np_shape)
        if label_np.shape[0] > image_np.shape[0]:
    
            z_difference= label_np.shape[0]-image_np.shape[0]
    
            top_slices= z_difference//2
    
            bottom_slices=z_difference-top_slices
    
            label_np= label_np[top_slices:-bottom_slices,:,:]
    
        elif(label_np.shape[0] < image_np.shape[0]):
    
            z_difference= image_np.shape[0]-label_np.shape[0]
    
            top_slices= z_difference//2
    
            bottom_slices=z_difference-top_slices
       
            image_np= image_np[top_slices:-bottom_slices,:,:]
       
        for i in range(label_np.shape[0]):
    
            if np.sum(label_np[i,:,:])>600000:
    
                annotated_slices.append(i)
    
        image_np = image_np[annotated_slices,:,:]
    
        label_np = label_np[annotated_slices,:,:]
        #aug_image = []
        #aug_mask = []
        for stack in range(image_np.shape[0]):
            image_slice = image_np[stack] 
            label_slice = label_np[stack]
            label_size = label_slice.shape[0]*label_slice.shape[1]
            label_slice_degenerated = label_slice[label_slice>=2]
            print(f"The length of pixel degenerated is {len(label_slice_degenerated)}")
            print(f"The size of the original label is {label_size}")
            fraction = len(label_slice_degenerated)/(label_size)
            print(f"The fraction of the ratio = {fraction}")
            if fraction > 0.10:
                image_tensor=torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)
                label_tensor=torch.tensor(label_slice,dtype=torch.float32).unsqueeze(0)

                seed = torch.randint(0,2**32,(1,)).item()
                torch.manual_seed(seed)
                transformed_image_slice=transform(image_tensor)
                torch.manual_seed(seed)
                transformed_label_slice=transform(label_tensor)
                os.makedirs(directory_path_image, exist_ok=True)
                os.makedirs(directory_path_label, exist_ok=True)
                image_name_slice = os.path.basename(image_name[:-4])+f"_augmented_slice_{stack}.tif"
                image_path = os.path.join(directory_path_image, image_name_slice)
                label_name_slice = os.path.basename(label_name[:-4])+f"_augmented_slice_{stack}.tif"
                label_path = os.path.join(directory_path_label, label_name_slice)
                tifffile.imwrite(image_path,transformed_image_slice.squeeze(0).cpu().numpy().astype(np.uint16))
                tifffile.imwrite(label_path, transformed_label_slice.squeeze(0).cpu().numpy().astype(np.uint16))
              
    
        for i in range((image_np.shape[0])):
            image_slice=image_np[i]
            label_slice=label_np[i]
            
            os.makedirs(directory_path_image, exist_ok=True)
            os.makedirs(directory_path_label, exist_ok=True)
            image_name_slice = os.path.basename(image_name[:-4])+f"slice_{i}.tif"
            image_path = os.path.join(directory_path_image, image_name_slice)
            label_name_slice = os.path.basename(label_name[:-4])+f"slice_{i}.tif"
            label_path = os.path.join(directory_path_label, label_name_slice)
            tifffile.imwrite(image_path,image_slice)
            tifffile.imwrite(label_path, label_slice)
    print("All_image and mask_slices saved")

def save_images(image_list,save_path):
    #directory_path=os.path.join(save_path, directory_name)
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(image_list)):
            image_name=image_list[i]
            image=tifffile.imread(image_name)
            image_path = os.path.join(save_path, os.path.basename(image_name))
            tifffile.imwrite(image_path, image)

            
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
        #return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy().squeeze(0), label.squeeze(0).cpu().detach().numpy().squeeze(0)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


TARGET_SIZE   = 1024   # images are resized to this before patching
PATCH_SIZE    = 256    # patchify patch size
STEP_SIZE     = 128    # patchify step (50 % overlap)
MODEL_SIZE    = 224    # size the model expects
GRID_N        = (TARGET_SIZE - PATCH_SIZE) // STEP_SIZE + 1   # = 7

SAVE_ROOT = "/projects/wan-lab/zebrafish/unet_attn_2d_implementation/Swim-unet_patch"

def test_single_volume_zebrafish(
    image, name, label, net, classes,
    patch_size=None,          # kept for API compatibility, not used
    test_save_path=None,
    test_plot_save_path=None,
):
    """
    Patch-based inference for a single zebrafish volume (stack of 2-D slices).

    For every slice:
      1. Resize to TARGET_SIZE × TARGET_SIZE
      2. Patchify → PATCH_SIZE × PATCH_SIZE patches with STEP_SIZE overlap
      3. Resize each patch to MODEL_SIZE for the network
      4. Resize predictions back to PATCH_SIZE
      5. Unpatchify → full TARGET_SIZE reconstruction
      6. Compute per-patch metrics  AND  whole-slice metrics

    Returns
    -------
    metric_list : list of (dice, hd95) per class, averaged over all slices
    patch_metric_list : list of per-patch (dice, hd95) per class, averaged
    """

    image = image.squeeze(0).cpu().detach().numpy().squeeze(0)
    label = label.squeeze(0).cpu().detach().numpy().squeeze(0)
    image, label = extract_annotated_image_test(image, label)

    if label.shape[0] != image.shape[0]:
        raise ValueError(
            f"Mismatch: Label stacks = {label.shape[0]}, "
            f"Image stacks = {image.shape[0]}"
        )

    n_slices     = image.shape[0]
    # Will hold reconstructed full-size predictions for every slice
    full_prediction = np.zeros_like(label)       # (S, H, W)  original label dims

    # Accumulate per-patch metrics across all slices
    all_patch_metrics = []   # each entry: list-of-(dice,hd95) per class for one patch

    colors = ["none", "green", "orange", "red"]
    cmap   = ListedColormap(colors)

    # -----------------------------------------------------------------------
    # Main loop over slices
    # -----------------------------------------------------------------------
    for ind in range(n_slices):
        raw_slice       = image[ind]          # original spatial dims
        raw_label_slice = label[ind]

        orig_h, orig_w  = raw_slice.shape
        orig_hl, orig_wl = raw_label_slice.shape

        # ── 1. Resize to TARGET_SIZE ────────────────────────────────────────
        img_1024   = zoom(raw_slice,       (TARGET_SIZE / orig_h,  TARGET_SIZE / orig_w),  order=3)
        label_1024 = zoom(raw_label_slice, (TARGET_SIZE / orig_hl, TARGET_SIZE / orig_wl), order=0)

        # ── 2. Patchify ─────────────────────────────────────────────────────
        img_patches_grid   = patchify(img_1024,   (PATCH_SIZE, PATCH_SIZE), step=STEP_SIZE)
        # shape: (GRID_N, GRID_N, PATCH_SIZE, PATCH_SIZE)
        label_patches_grid = patchify(label_1024, (PATCH_SIZE, PATCH_SIZE), step=STEP_SIZE)

        img_patches_flat   = img_patches_grid.reshape(-1, PATCH_SIZE, PATCH_SIZE)
        label_patches_flat = label_patches_grid.reshape(-1, PATCH_SIZE, PATCH_SIZE)

        # ── 3 & 4. Model inference per patch ────────────────────────────────
        pred_patches_flat = []
        net.eval()

        for patch_img, patch_lbl in zip(img_patches_flat, label_patches_flat):
            # Resize to model input size
            patch_model = zoom(patch_img, (MODEL_SIZE / PATCH_SIZE, MODEL_SIZE / PATCH_SIZE), order=3)
            inp = torch.from_numpy(patch_model).unsqueeze(0).unsqueeze(0).float().cuda()

            with torch.no_grad():
                out  = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
                pred = out.cpu().detach().numpy()   # (MODEL_SIZE, MODEL_SIZE)

            # Resize prediction back to PATCH_SIZE (nearest-neighbour – it's a label map)
            pred_256 = zoom(pred, (PATCH_SIZE / MODEL_SIZE, PATCH_SIZE / MODEL_SIZE), order=0)
            pred_patches_flat.append(pred_256)

            # Per-patch metrics vs the corresponding label patch
            patch_metrics = [
                calculate_metric_percase(pred_256 == cls, patch_lbl == cls)
                for cls in range(1, classes)
            ]
            all_patch_metrics.append(patch_metrics)

        # ── 5. Unpatchify → full 1024 × 1024 prediction ─────────────────────
        pred_grid = np.array(pred_patches_flat).reshape(
            GRID_N, GRID_N, PATCH_SIZE, PATCH_SIZE
        )
        recon_1024 = unpatchify(pred_grid, (TARGET_SIZE, TARGET_SIZE))

        # Resize reconstruction back to original label dimensions
        recon_orig = zoom(recon_1024, (orig_hl / TARGET_SIZE, orig_wl / TARGET_SIZE), order=0)
        full_prediction[ind] = recon_orig

        # ── 6. Optional: per-slice visualisation ────────────────────────────
        if test_plot_save_path is not None:
            os.makedirs(test_plot_save_path, exist_ok=True)

            slice_metrics = [
                calculate_metric_percase(recon_orig == cls, raw_label_slice == cls)
                for cls in range(1, classes)
            ]

            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            axes[0].imshow(raw_slice,       cmap="gray");               axes[0].set_title(f"Image Slice {ind}");      axes[0].axis("off")
            axes[1].imshow(raw_slice,       cmap="gray")
            axes[1].imshow(raw_label_slice, cmap=cmap, alpha=0.5, vmin=0, vmax=len(colors) - 1)
            axes[1].set_title(f"Label Slice {ind}");                    axes[1].axis("off")
            axes[2].imshow(raw_slice,       cmap="gray")
            axes[2].imshow(recon_orig,      cmap=cmap, alpha=0.5, vmin=0, vmax=len(colors) - 1)
            axes[2].set_title(f"Prediction Slice {ind}");               axes[2].axis("off")

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.22)
            fig.text(
                0.05, 0.02,
                f"Dice – Healthy={slice_metrics[0][0]:.4f}  "
                f"Dystrophic={slice_metrics[1][0]:.4f}  "
                f"Necrotic={slice_metrics[2][0]:.4f}",
                fontsize=11, color="black",
            )
            plt.savefig(os.path.join(test_plot_save_path, f"{name[0]}_slice{ind}.png"))
            plt.close()

    # -----------------------------------------------------------------------
    # Whole-volume metrics (over the fully reconstructed prediction)
    # -----------------------------------------------------------------------
    metric_list = [
        calculate_metric_percase(full_prediction == cls, label == cls)
        for cls in range(1, classes)
    ]

    # Average per-patch metrics across all patches in this volume
    all_patch_metrics = np.array(all_patch_metrics)          # (n_patches_total, n_classes, 2)
    patch_metric_avg  = all_patch_metrics.mean(axis=0).tolist()  # (n_classes, 2)

    # ── Save full prediction TIFF ────────────────────────────────────────────
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        tifffile.imwrite(os.path.join(test_save_path, name[0]), full_prediction.astype(np.uint8))

    return metric_list, patch_metric_avg


# def test_single_volume_zebrafish(image, name, label, net, classes, patch_size=[256, 256], test_save_path=None, test_plot_save_path=None):
    
#     image, label = image.squeeze(0).cpu().detach().numpy().squeeze(0), label.squeeze(0).cpu().detach().numpy().squeeze(0)
    
#     image, label = extract_annotated_image_test(image, label)
   
#     if label.shape[0] != image.shape[0]:
#         raise ValueError(f"Mismatch : Label Stacks = { label.shape[0]}, Image Stacks = {image.shape[0]}")
#     if len(image.shape) == 3:
        
#         prediction = np.zeros_like(label)
        
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             label_slice = label[ind,:,:]
#             x, y = slice.shape[0], slice.shape[1]
#             x_l,y_l = label_slice.shape[0], label_slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 #print(out.shape)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     #pred = zoom(out, (1, x_l / patch_size[0], y_l / patch_size[1]), order=0)
#                     pred = zoom(out, (1, x_l / patch_size[0], y_l / patch_size[1]), order=3)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))

#     if test_save_path is not None:
#         os.makedirs(test_save_path, exist_ok=True)
#         print(f"The test save path = {test_save_path}")
#         print(f"The name = {name}")
#         prediction_path = os.path.join(str(test_save_path), name[0])
#         tifffile.imwrite(prediction_path, prediction)
#     if test_plot_save_path is not None:
        
#         os.makedirs(test_plot_save_path, exist_ok=True)
#         colors = ['none', 'green', 'orange', 'red' ]
#         cmap = ListedColormap(colors)
#         for i ,ind in enumerate(range(image.shape[0])):
#             metric_list_per_slice = []
#             for j in range(1, classes):
#                 metric_list_per_slice.append(calculate_metric_percase(prediction[ind,:,:]==j,label[ind,:,:]==j))
#             print(f"The loss of per label per index = {metric_list_per_slice}")
#             fig, axes=plt.subplots(1,3,figsize=(16,6))
#             axes[0].imshow(image[ind,:,:], cmap='gray')
#             axes[0].set_title(f'Image Slice {ind}')
#             axes[0].axis('off')

#             axes[1].imshow(image[ind,:,:], cmap='gray')
#             #axes[1].imshow(label[ind,:,:], cmap='gray')
#             axes[1].imshow(label[ind,:,:], cmap=cmap, alpha=0.5, vmin=0, vmax=len(colors)-1)
#             axes[1].set_title(f'Label Slice {ind}')
#             axes[1].axis('off')

#             axes[2].imshow(image[ind,:,:], cmap='gray')
#             #axes[2].imshow(prediction[ind,:,:], cmap='gray')
#             axes[2].imshow(prediction[ind,:,:], cmap=cmap, alpha=0.5, vmin=0, vmax=len(colors)-1)
#             axes[2].set_title(f'Prediction Slice {ind}')
#             axes[2].axis('off')
#             plot_filename=os.path.join(str(test_plot_save_path), f"{name[0]}_slice{ind}.png")
#             plt.tight_layout()
#             plt.subplots_adjust(bottom=0.20)
            
#             fig.text(0.05, 0.02,f'The predicticted Dice Score for Fine = {metric_list_per_slice[0][0]}\n'
#             f'The predicticted Dice Score for Bad = {metric_list_per_slice[1][0]}\n'
#             f'The predicticted Dice Score for Degenerated = {metric_list_per_slice[2][0]}',fontsize = 12, color='black')
#             #fig.text(0.05, 0.02,f'The predicticted Dice Score for Bad =  ,{metric_list_per_slice[1]}',fontsize = 12, color='black')
#             #fig.text(0.05, 0.02,f'The predicticted Dice Score for Degenerated =  ,{metric_list_per_slice[2]}',fontsize = 12, color='black')
#             #plt.tight_layout()
#             plt.savefig(plot_filename)
#             plt.close()
#     #print(f"The metric list for Image {name} is {metric_list} ")
#     return metric_list



