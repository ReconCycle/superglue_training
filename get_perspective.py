import os
import numpy as np
import cv2
import time
from utils.preprocess_utils import get_perspective_mat
from tqdm import tqdm
import yaml
from pathlib import Path


np.random.seed(100) # changed the seed if needed
# image_dir = "assets/outdoor_test_images" 

with open("configs/coco_config.yaml", 'r') as file:
    config = yaml.full_load(file)

dataset_params = config["dataset_params"]

image_dir = Path(dataset_params["dataset_path"]) / "images"

aug_params = dataset_params['augmentation_params']

txt_file = open("assets/my_homo.txt", 'w') #path where the generated homographies should be stored
image_save_path = "assets/gen_homos" #path where the original and warped image will be stored for visualization
if not os.path.isdir(image_save_path): os.makedirs(image_save_path)
content = os.listdir(image_dir)
ma_fn = lambda x: float(x)
for kk, i in tqdm(list(enumerate(content))):
    if not os.path.splitext(i)[-1] in [".jpg", ".png"]:
        continue
    image = cv2.imread(os.path.join(image_dir, i))
    height, width = image.shape[0:2]
    #all the individual perspective component range should be adjusted below
    # homo_matrix = get_perspective_mat(0.85,center_x=width//2, center_y=height//2, pers_x=0.0008, pers_y=0.0008, shear_ratio=0.04, shear_angle=10, rotation_angle=25, scale=0.6, trans=0.6)
    homo_matrix = get_perspective_mat(aug_params['patch_ratio'], width//2, height//2, aug_params['perspective_x'], aug_params['perspective_y'], aug_params['shear_ratio'], aug_params['shear_angle'], aug_params['rotation_angle'], aug_params['scale'], aug_params['translation'])
    res_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
    txt_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(i, *list(map(ma_fn, list(homo_matrix.reshape(-1))))))
    write_img = np.concatenate([image, res_image], axis=1)
    cv2.imwrite(os.path.join(image_save_path, "{}.png".format(kk+1)), write_img)