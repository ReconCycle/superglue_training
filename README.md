# Notes Seb

`get_perspective_coco.py` generates perspecives from coco_config

# Debugging with `match_homograph.py` on tiny dataset `output_2023-11-10_100` with:
```
python3 match_homography.py --eval --superglue indoor --input_homography assets/homo_output_2023-11-10_100.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-05-23_synthetic_dataset/output_2023-11-10_100
```
compare with:
```
python3 match_homography.py --eval --superglue output/train/default/weights/best.pt --input_homography assets/homo_output_2023-11-10_100.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-05-23_synthetic_dataset/output_2023-11-10_100
```

# Evaluation on `output_2023-09-25_40000` val set with:

This IS using my 360deg rotations!


NOTE: we limit to 500 otherwise it takes a while

{indoor, outdoor, coco_homo, output/train/default/weights/best.pt}

## output/train/default/weights/best.pt
```
python3 match_homography.py --eval --superglue output/train/default/weights/best.pt --input_homography assets/homo_output_2023-09-25_40000.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-05-23_synthetic_dataset/output_2023-09-25_40000 --max_length 500
```
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall
0.65     2.07    5.48    67.78   79.78
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall
23.38    34.99   51.73   67.78   79.78


## indoor
```
python3 match_homography.py --eval --superglue indoor --input_homography assets/homo_output_2023-09-25_40000.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-05-23_synthetic_dataset/output_2023-09-25_40000 --max_length 500
```
Evaluation Results (mean over 500 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall
0.25     0.65    2.70    48.67   62.51
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall
16.36    27.34   42.90   48.67   62.51

## outdoor
```
python3 match_homography.py --eval --superglue outdoor --input_homography assets/homo_output_2023-09-25_40000.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-05-23_synthetic_dataset/output_2023-09-25_40000 --max_length 500
```
Evaluation Results (mean over 500 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall
0.24     0.78    2.94    38.18   45.21
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall
10.93    17.69   26.45   38.18   45.21

# Evaluation on `coco_test_images` COCO2017 test set (Mean over 199 pairs):

This ISN'T using my 360deg rotations!

## output/train/default/weights/best.pt

```
python3 match_homography.py --eval --superglue output/train/default/weights/best.pt --input_homography assets/coco_test_images_homo.txt --input_dir assets/coco_test_images
```

```
Evaluation Results (mean over 199 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall
0.91     2.49    6.18    75.44   79.53
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall
34.66    51.72   70.58   75.44   79.53
```

## indoor

```
python3 match_homography.py --eval --superglue indoor --input_homography assets/coco_test_images_homo.txt --input_dir assets/coco_test_images
```
Evaluation Results (mean over 199 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall
0.00     0.38    2.60    73.35   94.59
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall
30.22    47.72   68.11   73.35   94.59

## outdoor

```
python3 match_homography.py --eval --superglue outdoor --input_homography assets/coco_test_images_homo.txt --input_dir assets/coco_test_images
```
Evaluation Results (mean over 199 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall
0.31     1.49    6.99    89.12   99.65
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall
32.19    49.15   71.56   89.12   99.65

# Evaluation on `2023-02-20_hca_backs_processed`

This IS using my 360deg rotations!

## output/train/default/weights/best.pt

```
python3 match_homography.py --eval --superglue output/train/default/weights/best.pt --input_homography assets/homo_2023-02-20_hca_backs_processed.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed
```
Evaluation Results (mean over 453 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.00     0.00    1.07    80.54   97.44  
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
4.31     19.64   54.72   80.54   97.44  
angles:
med      mean    std    
0.00553  0.0108  0.0213


## indoor
```
python3 match_homography.py --eval --superglue indoor --input_homography assets/homo_2023-02-20_hca_backs_processed.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed
```
Evaluation Results (mean over 453 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.00     0.00    0.00    59.38   67.65  
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
4.84     13.27   32.09   59.38   67.65  
angles:
med      mean    std    
0.0117   0.191   0.458


## outdoor
```
python3 match_homography.py --eval --superglue outdoor --input_homography assets/homo_2023-02-20_hca_backs_processed.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed
```
Evaluation Results (mean over 453 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.00     0.00    0.00    39.95   41.79  
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.58     5.54    17.05   39.95   41.79  
angles:
med      mean    std    
0.685    1.08    1.1 


## coco_homo
```
python3 match_homography.py --eval --superglue coco_homo --input_homography assets/homo_2023-02-20_hca_backs_processed.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed
```
Evaluation Results (mean over 453 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.00     0.00    0.24    28.98   30.17  
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
3.65     6.66    15.63   28.98   30.17  
angles:
med      mean    std    
0.81     1.03    0.98


# Evaluation on `2023-12-04_hcas_fire_alarms_sorted_cropped`

## output/train/2023-11-18_superglue_model/weights/best.pt

```
python3 match_homography.py --eval --superglue output/train/2023-11-18_superglue_model/weights/best.pt --input_homography assets/2023-12-04_hcas_fire_alarms_sorted_cropped/homo.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped
```

```
Evaluation Results (mean over 1632 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.47     2.03    7.10    78.78   91.20  
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
34.62    54.15   75.57   78.78   91.20  
angles:
med      mean    std    
0.00245  0.00741         0.0684 
```

## Now with augmentations:

```
python3 match_homography.py --eval --superglue output/train/2023-11-18_superglue_model/weights/best.pt --input_homography assets/2023-12-04_hcas_fire_alarms_sorted_cropped/homo.txt --input_dir /home/sruiz/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped --apply_aug
```

```
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
0.63     2.05    7.15    69.90   nan    
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
27.77    47.79   70.62   69.90   nan    
angles:
med      mean    std    
0.00313  0.0153  0.124 
```

#  SuperGlue - Reimplementation
This repository contains training and evaluation code for Superglue model with homography pairs generated from COCO dataset. The code is adapted from the inference only [official implementation](https://github.com/magicleap/SuperGluePretrainedNetwork) released by MagicLeap

  

##  Requirements
- torch>=1.8.1
- torch_scatter>=2.06
- matplotlib>=3.1.3
- opencv-python==4.1.2.30
- numpy>=1.18.1
- PyYAML>=5.4.1
- wandb (If logging of checkpoints to cloud is need)
- albumentations 

##  Training data
COCO 2017 dataset is used for training. Random homographies are generated at every iteration and matches are computed using the know homography matrix. Download the 'train2017', 'val2017', and 'annotations' folder of COCO 2017 dataset and put that path in the config file used for training.

##  Training

All the parameters of training are provided in the coco_config.yaml in the configs folder. Change that file inplace and start the training. Or clone that file to make the changes and mention the custom config path in training command. Parameters of training are explained in comments in the coco_config.yaml file. To start the training run,

  

    python3 train_superglue.py --config_path configs/coco_config.yaml

Incase of Multi-GPU training, distributed setting is used. So run the following command,

  

    python3 -m torch.distributed.launch --nproc_per_node="NUM_GPUS" train_superglue.py --config_path configs/coco_config.yaml

Only singe-node training is supported as of now.

Checkpoints are saved at the end of every epoch, and best checkpoint is determined by weighted score of AUC at different thresholds, precision and recall computed on COCO 2017 val dataset using random homographies. Validation score is computed on fixed set of images and homographies for consistency across runs. Image and homography info used for validation is present at assets/coco_val_images_homo.txt

  
##  Evaluation

The official implementation has evaluation code for testing on small set of scannet scene pairs. Since our model in trained with random homographies, evaluating on scenes with random 3D camera movements doesn't perform well as pretrained indoor model. Instead we evaluate on test images of COCO, indoor and outdoor dataset(https://dimlrgbd.github.io/) with random homographies. Images are selected from the datasets and random homographies are generated for each of them. Based on matches given by the model, we determine the homography matrix using DLT and RANSAC implementation. As mentioned in paper, we report the AUC at 5, 10, 25 thresholds(for corner points), precision and recall. For evaluation run the following command,

  

    python3 match_homography.py --eval --superglue coco_homo

Parameter --superglue determines the checkpoint used and should be one of the following,

  

- Use **coco_homo** to run with the released coco homography model

- Use **PATH_TO_YOUR_.PT** to run with your trained model

- Use **indoor** to run with official indoor pretrained model

- Use **outdoor** to run with official outdoor pretrained model

  

Add --viz flag to dump the matching info image to 'dump_homo_pairs' folder.

If you want to evaluate with scannet pairs, run the above command with match_pairs.py with same parameters

##  Evaluation Results

Following are the results on three different sets of COCO2017 test images, indoor images and outdoor images using randomly generated homographies at assets/coco_test_images_homo.txt, assets/indoor_test_images_homo.txt, assets/outdoor_test_images_homo.txt respectively.

###  Homography using RANSAC 
#### Indoor test set (Mean over 144 pairs)

    python3 match_homography.py --eval --superglue {indoor, outdoor, coco_homo, ckpt_path} --input_homography assets/indoor_test_images_homo.txt --input_dir assets/indoor_test_images

| Models | AUC@5 | AUC@10 | AUC@25 | Prec | Recall |
|--|--|--|--|--|--|
| Indoor |26.44  | 37.71 | 52.66 | 62.14 | 88.28
| Outdoor |28.01  | 40.38 | 56.43 | 81.74 | 98.02
| COCO_homo |28.41  | 42.45 | 57.90 | 68.70 | 87.56
#### Outdoor test set (Mean over 145 pairs)

    python3 match_homography.py --eval --superglue {indoor, outdoor, coco_homo, ckpt_path} --input_homography assets/outdoor_test_images_homo.txt --input_dir assets/outdoor_test_images

| Models | AUC@5 | AUC@10 | AUC@25 | Prec | Recall |
|--|--|--|--|--|--|
| Indoor |23.41  | 35.08 | 46.82 | 54.07 | 79.70
| Outdoor |23.92  | 33.26 | 47.57 | 76.44 | 96.07
| COCO_homo |23.82  | 33.49 | 46.69 | 53.90 | 70.18
#### COCO2017 test set (Mean over 199 pairs)

    python3 match_homography.py --eval --superglue {indoor, outdoor, coco_homo, ckpt_path} --input_homography assets/coco_test_images_homo.txt --input_dir assets/coco_test_images

| Models | AUC@5 | AUC@10 | AUC@25 | Prec | Recall |
|--|--|--|--|--|--|
| Indoor |28.55  | 47.18 | 69.04 | 75.31 | 95.71
| Outdoor |32.40  | 50.20 | 71.10 | 88.61 | 99.71
| COCO_homo |34.06  | 52.33 | 73.03 | 80.87 | 96.22


### Homography using DLT(Most confident 4 matches)
#### Indoor test set(Mean over 144 pairs)
| Models | AUC@5 | AUC@10 | AUC@25 | Prec | Recall |
|--|--|--|--|--|--|
| Indoor |0.57  | 1.07 | 3.43 | 62.14 | 88.28
| Outdoor |0.38  | 1.17 | 4.45 | 81.74 | 98.02
| COCO_homo |1.13  | 3.13 | 7.95 |68.70| 87.56
#### Outdoor test set (Mean over 145 pairs)
| Models | AUC@5 | AUC@10 | AUC@25 | Prec | Recall |
|--|--|--|--|--|--|
| Indoor |0.89  | 2.27 | 4.36 | 54.07 | 79.70
| Outdoor |0.77  | 2.17 | 6.38 | 76.44 | 96.06
| COCO_homo |0.45  | 1.89 | 5.60 | 53.90 | 70.18
#### COCO2017 test set (Mean over 199 pairs)
| Models | AUC@5 | AUC@10 | AUC@25 | Prec | Recall |
|--|--|--|--|--|--|
| Indoor |0.31  | 1.01 | 3.05 | 75.31 | 95.71
| Outdoor |0.29  | 1.41 | 8.10 | 88.61 | 99.71
| COCO_homo |1.28  | 4.65 | 10.62 | 80.87 | 96.22

Output of COCO_homo model on few of COCO2017 test images
<img src="assets/coco_homo_teaser.gif" width="560">

##  Creating you own homography test set
Incase you want to generate your own test set with random homographies that is compatible with match_homography.py run the command `python3 get_perspective.py`.

All the parameters regarding the random homography range, input and output folder are mentioned in the file itself. While running the match_homography.py you should mention the additional parameters --input_homography and --input_dir that points to the generated homography text file and images directory respectively.

## Credits
- https://github.com/magicleap/SuperGluePretrainedNetwork - Training code is adapted from this official repo
- https://github.com/ultralytics/yolov5 - Training code template is inspired from this repo