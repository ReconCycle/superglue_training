import os
import torch
import numpy as np
import cv2
import pycocotools.coco as coco
from torch.utils.data import Dataset
from .preprocess_utils import get_perspective_mat, scale_homography, resize_aspect_ratio
from pathlib import Path

class COCO_loader(Dataset):
    def __init__(self, dataset_params, typ="train"):
        super(COCO_loader, self).__init__()
        self.typ = typ


        self.config = dataset_params
        self.aug_params = dataset_params['augmentation_params']
        self.dataset_path = dataset_params['dataset_path']
        self.aspect_resize = dataset_params['resize_aspect']

        if typ == "train":
            self.apply_aug = dataset_params['apply_color_aug']
        else:
            self.apply_aug = False

        self.images_path = self.dataset_path
        self.json_path = os.path.join(self.dataset_path, 'coco', '{}.json'.format(typ))
        self.coco_json = coco.COCO(self.json_path)
        self.images = self.coco_json.getImgIds()
        if self.apply_aug:
            import albumentations as alb
            self.aug_list = [# alb.OneOf([alb.RandomBrightness(limit=0.4, p=0.6), alb.RandomContrast(limit=0.3, p=0.7)], p=0.6),
                             alb.OneOf([alb.MotionBlur(p=0.5), alb.GaussNoise(p=0.6)], p=0.5),
                             alb.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.8), #! added by Seb
                            #  alb.RandomBrightnessContrast(p=0.5),
                            alb.ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, always_apply=None, p=0.8)
                             ]
            self.aug_func = alb.Compose(self.aug_list, p=1.0)

    def __len__(self):
        return len(self.images)

    def apply_augmentations(self, image1, image2):
        image1_dict = {'image': image1}
        image2_dict = {'image': image2}
        result1, result2 = self.aug_func(**image1_dict), self.aug_func(**image2_dict)
        return result1['image'], result2['image']

    def __getitem__(self, index: int):
        resize = True
        img_id = self.images[index]
        file_name = self.coco_json.loadImgs(ids=[img_id])[0]['file_name']
        file_path = os.path.join(self.images_path, file_name)

        image = cv2.imread(file_path) #! seb: used to load as: cv2.IMREAD_GRAYSCALE
        if self.aspect_resize:
            image = resize_aspect_ratio(image, self.config['image_height'], self.config['image_width'])
            resize = False
        height, width = image.shape[0:2]

        if self.typ == "val":
            # for validation, fix the seed for each image, so that we get the same homo_matrix every time

            # get the initial state of the RNG
            st0 = np.random.get_state()

            np.random.seed(img_id)

        homo_matrix, angle = get_perspective_mat(self.aug_params['patch_ratio'], width//2, height//2, self.aug_params['perspective_x'], self.aug_params['perspective_y'], self.aug_params['shear_ratio'], self.aug_params['shear_angle'], self.aug_params['rotation_angle'], self.aug_params['scale'], self.aug_params['translation'])

        if self.typ == "val":
            # set the state back to what it was originally
            np.random.set_state(st0)

        warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
        if resize:
            orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
            warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
        else:
            orig_resized = image
            warped_resized = warped_image
        if self.apply_aug:
            orig_resized, warped_resized = self.apply_augmentations(orig_resized, warped_resized)

        # ! seb: now convert to grayscale
        orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
        warped_resized = cv2.cvtColor(warped_resized, cv2.COLOR_BGR2GRAY)

        homo_matrix = scale_homography(homo_matrix, height, width, self.config['image_height'], self.config['image_width']).astype(np.float32)
        orig_resized = np.expand_dims(orig_resized, 0).astype(np.float32) / 255.0
        warped_resized = np.expand_dims(warped_resized, 0).astype(np.float32) / 255.0
        return orig_resized, warped_resized, homo_matrix

# class COCO_valloader(Dataset):
#     def __init__(self, dataset_params):
#         super(COCO_valloader, self).__init__()

#         #! Run get_prespective.py first!

#         self.config = dataset_params
#         self.dataset_path = dataset_params['dataset_path']
#         self.images_path = os.path.join(self.dataset_path, "val2017_COCO_valloader")
#         # self.images_path = os.path.join(self.dataset_path, "val2017")
#         # self.images_path = self.dataset_path
#         self.txt_path = str(Path(__file__).parent.parent / 'assets/coco_val_images_homo.txt')
#         print("txt_path", self.txt_path)
#         with open(self.txt_path, 'r') as f:
#             self.image_info = f.readlines()

        

#     def __len__(self):
#         return len(self.image_info)

#     def __getitem__(self, index: int):
#         split_info = self.image_info[index].strip().split(' ')
#         image_name = split_info[0]
#         homo_info = list(map(lambda x: float(x), split_info[1:]))
#         homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
#         image = cv2.imread(os.path.join(self.images_path, image_name), cv2.IMREAD_GRAYSCALE)
#         height, width = image.shape[0:2]
#         warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
#         orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
#         warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
#         homo_matrix = scale_homography(homo_matrix, height, width, self.config['image_height'], self.config['image_width']).astype(np.float32)
#         orig_resized = np.expand_dims(orig_resized, 0).astype(np.float32) / 255.0
#         warped_resized = np.expand_dims(warped_resized, 0).astype(np.float32) / 255.0
#         return orig_resized, warped_resized, homo_matrix

def collate_batch(batch):
    list_elem = list(zip(*batch))
    orig_resized = torch.stack([torch.from_numpy(i) for i in list_elem[0]], 0)
    warped_resized = torch.stack([torch.from_numpy(i) for i in list_elem[1]], 0)
    homographies = torch.stack([torch.from_numpy(i) for i in list_elem[2]], 0)
    orig_warped_resized = torch.cat([orig_resized, warped_resized], 0)
    return [orig_warped_resized, homographies]



