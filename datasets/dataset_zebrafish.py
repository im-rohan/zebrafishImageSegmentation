import os
import random
import numpy as np
import torch
import tifffile
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import extract_annotated_image_train, extract_annotated_image_train_augment,extract_annotated_image_test, load_files, save_images
from torchvision import transforms

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        #print(image.shape)
        #print(label.shape)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        x_l,y_l = label.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x_l, self.output_size[1] / y_l), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        #print(f"Image shape = {image.shape}")
        #print(f"Label shape = {label.shape}")
        sample = {'image': image, 'label': label.long()}
        return sample



class Zebrafish_dataset(Dataset):
    def __init__(self, base_dir, mode='train',transform_augment = None, transform=None):
        self.transform=transform # using transfrom in torch
        self.transform_augment = transform_augment
        self.mode=mode
        image_path_3d=os.path.join(base_dir, 'images')
        #print(image_path_3d)
        label_path_3d=os.path.join(base_dir, '16_bit_mask')
        image_path_2d_train=os.path.join(base_dir, "train_images_2d")
        label_path_2d_train = os.path.join(base_dir, "train_labels_2d")
        image_path_2d_val=os.path.join(base_dir, "val_images_2d")
        label_path_2d_val= os.path.join(base_dir, "val_labels_2d")
        image_path_3d_test = os.path.join(base_dir, "test_images_3d")
        label_path_3d_test = os.path.join(base_dir, "test_labels_3d")


        if self.mode == 'test':
            image_path_2d_train_augmented=os.path.join(base_dir, "train_images_2d_augmented")
            label_path_2d_train_augmented=os.path.join(base_dir, "train_labels_2d_augmented")
            image_path_2d_val_augmented=os.path.join(base_dir, "val_images_2d_augmented")
            label_path_2d_val_augmented=os.path.join(base_dir, "val_labels_2d_augmented")
        else:
            image_path_2d_train_augmented=os.path.join(base_dir, "train_images_2d_augmented_patches")
            label_path_2d_train_augmented=os.path.join(base_dir, "train_labels_2d_augmented_patches")
            image_path_2d_val_augmented=os.path.join(base_dir, "val_images_2d_augmented_patches")
            label_path_2d_val_augmented=os.path.join(base_dir, "val_labels_2d_augmented_patches")
    
        '''
        if not os.path.exists(image_path_2d_train) or not os.path.exists(label_path_2d_train):
            image_files = load_files(image_path_3d)
            label_files = load_files(label_path_3d)
           
            train_images, test_images, train_labels, test_labels=train_test_split(image_files, label_files, test_size=0.25, random_state=42, shuffle=True)
            train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.20, random_state=42, shuffle=True) 
            
            extract_annotated_image_train(train_images, train_labels,image_path_2d_train, label_path_2d_train)
            extract_annotated_image_train(val_images, val_labels,image_path_2d_val, label_path_2d_val )
            save_images(test_images, image_path_3d_test) 
            save_images(test_labels, label_path_3d_test)
        '''
        if not os.path.exists(image_path_2d_train_augmented) or not os.path.exists(label_path_2d_train_augmented):
            image_files = load_files(image_path_3d)
            label_files = load_files(label_path_3d)
           
            train_images, test_images, train_labels, test_labels=train_test_split(image_files, label_files, test_size=0.25, random_state=42, shuffle=True)
            train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.20, random_state=42, shuffle=True) 
            
            extract_annotated_image_train_augment(train_images, train_labels, self.transform_augment,image_path_2d_train_augmented, label_path_2d_train_augmented)
            extract_annotated_image_train_augment(val_images, val_labels,self.transform_augment,image_path_2d_val_augmented, label_path_2d_val_augmented )
            
            save_images(test_images, image_path_3d_test) 
            save_images(test_labels, label_path_3d_test)
        
        if self.mode == 'train':
            #self.image_path=image_path_2d_train
            #self.label_path=label_path_2d_train
            self.image_path=image_path_2d_train_augmented
            self.label_path=label_path_2d_train_augmented
        if self.mode == 'val':
            #self.image_path=image_path_2d_val
            #self.label_path=label_path_2d_val
            self.image_path=image_path_2d_val_augmented
            self.label_path=label_path_2d_val_augmented
            
        if self.mode=='test':
            self.image_path= image_path_3d_test    
            self.label_path =  label_path_3d_test
        
        self.image_files = sorted(os.listdir(self.image_path))
        self.label_files = sorted(os.listdir(self.label_path))

        if len(self.image_files) != len(self.label_files):
            raise ValueError(f"Mismatch: Length of image files  =  {len(self.image_files)} is not equal to length of label files, {len(self.label_files)}")

        self.data_size = len(self.image_files)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
       
        image_path=os.path.join(self.image_path,self.image_files[idx])
        #print(image_path)
        label_path=os.path.join(self.label_path, self.label_files[idx])
        #print(label_path)
        image=tifffile.imread(image_path)
        
        #image=image.unsqueeze(0)          
        label = tifffile.imread(label_path)
        
        #label=label.unsqueeze(0)
        sample = {'image':image, 'label':label}
    
        image_path=os.path.join(self.image_path,self.image_files[idx])
        #print(image_path)
        label_path=os.path.join(self.label_path, self.label_files[idx])
        #print(label_path)
        image=tifffile.imread(image_path)
        
        #image=image.unsqueeze(0)    
        label = tifffile.imread(label_path)
        
        #label=label.unsqueeze(0)
        sample = {'image':image, 'label':label}
        
        if self.mode=='test':
            
            #image=image.astype(np.float32)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            #label=label.astype(np.float32)
            label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
            sample = {'image':image, 'label':label, 'name':os.path.basename(image_path) }
        if self.transform:
            sample=self.transform(sample)
        return sample
