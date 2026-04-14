from patchify import patchify, unpatchify
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def create_patches(img_path, msk_path, patch_img_path, patch_msk_path, patch_size=256, step_size=128, target = 1024):
    for img, msk in zip(img_path, msk_path):
        image=tifffile.imread(img)
        mask=tifffile.imread(msk)
        
        h, w = image.shape
        h_l, w_l = mask.shape

        image = zoom(image, (target / h, target / w), order=3)
        mask = zoom(mask, (target / h_l, target / w_l), order=0)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        image_patches = patchify(image_np, (patch_size, patch_size), step=step_size)
        mask_patches = patchify(label_np, (patch_size, patch_size), step=step_size)
        
        image_patches = image_patches.reshape(-1, patch_size, patch_size)
        mask_patches = mask_patches.reshape(-1, patch_size, patch_size)
        
        save_patches(image_patches, patch_img_path, img)
        save_patches(mask_patches, patch_msk_path, msk)


def save_patches(patches, patch_path, image_path):
    for i, patch in enumerate(patches):
        image_name = os.path.basename(image_path).split(".")[0]
        image_name = image_name + "_patch" + str(i) + ".tif"
        save_loc = os.path.join(patch_path, image_name)
        tifffile.imwrite(save_loc, patch)