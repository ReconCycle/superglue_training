import os, sys
import numpy as np
import cv2
import time
import yaml
from utils.preprocess_utils import get_perspective_mat
from rich import print
import click
import shutil
from tqdm import tqdm
import albumentations as alb


def main():
    np.random.seed(100) # changed the seed if needed

    #! we iterate over all subdirectories of this and get the path
    image_dir = "/home/sruiz/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped" 
    save_path = "assets/2023-12-04_hcas_fire_alarms_sorted_cropped" #path where the original and warped image will be stored for visualization
    save_images = True

    
    if os.path.isdir(save_path) and os.listdir(save_path):
        if click.confirm(f"Do you want to delete {save_path}?", default=True):
            shutil.rmtree(save_path)
        else:
            sys.exit()
            return

    if not os.path.isdir(save_path): os.makedirs(save_path)

    with open(os.path.join(save_path, "homo.txt"), 'w') as txt_file: #path where the generated homographies should be stored
    
        with open("configs/get_perspective_hcas_firealarms_only.yaml", 'r') as file:
            config = yaml.full_load(file)

        dataset_params = config["dataset_params"]
        aug_params = dataset_params['augmentation_params']

        apply_aug = dataset_params['apply_color_aug'] #! usually only for training
        print("apply_aug", apply_aug)
        
        aug_list = [alb.OneOf([alb.MotionBlur(p=0.5), alb.GaussNoise(p=0.6)], p=0.5),
                    alb.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
                    alb.RandomBrightnessContrast(p=0.5),
                    ]
        aug_func = alb.Compose(aug_list, p=0.65)

        def apply_augmentations(image1, image2):
            image1_dict = {'image': image1}
            image2_dict = {'image': image2}
            result1, result2 = aug_func(**image1_dict), aug_func(**image2_dict)
            return result1['image'], result2['image']
        
        # content = os.listdir(image_dir)
        content = []
        for dirpath, dirnames, filenames in os.walk(image_dir):
            rel_dir = os.path.relpath(dirpath, image_dir)
            for filename in [f for f in filenames if f.endswith(".jpg") or f.endswith(".png")]:
                img_path = os.path.join(rel_dir, filename)
                content.append(img_path)
                print("img path", img_path)

        ma_fn = lambda x: float(x)
        for kk, i in enumerate(tqdm(content)):
            if not os.path.splitext(i)[-1] in [".jpg", ".png"]:
                continue
            image = cv2.imread(os.path.join(image_dir, i))
            height, width = image.shape[0:2]
            #all the individual perspective component range should be adjusted below
            homo_matrix = get_perspective_mat(aug_params['patch_ratio'], width//2, height//2, aug_params['perspective_x'], aug_params['perspective_y'], aug_params['shear_ratio'], aug_params['shear_angle'], aug_params['rotation_angle'], aug_params['scale'], aug_params['translation'])
            res_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
            if apply_aug:
                image, res_image = apply_augmentations(image, res_image)
                
            txt_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(i, *list(map(ma_fn, list(homo_matrix.reshape(-1))))))
            if save_images:
                write_img = np.concatenate([image, res_image], axis=1)
                cv2.imwrite(os.path.join(save_path, "{}.png".format(kk+1)), write_img)

if __name__ == '__main__':
    main()