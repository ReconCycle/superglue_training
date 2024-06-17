from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
import sys
import math
import logging
from datetime import datetime
from models.matching import Matching
from utils.common import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, frame2tensor, read_image,read_image_with_homography,
                          rotate_intrinsics, rotate_pose_inplane, compute_pixel_error,
                          scale_intrinsics, weights_mapping, download_base_files, download_test_images)
from utils.preprocess_utils import torch_find_matches
from rich import print
import albumentations as alb

from data_loader import DataLoader, random_seen_unseen_class_split
import exp_utils as exp_utils
from exp_utils import str2bool



torch.set_grad_enabled(False)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # for dataloader;
    parser.add_argument(
        '--img_path', type=str, default='~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped',
        help='image directory for dataloader')
    
    parser.add_argument(
        '--batch_size', type=int, default=56, help='Dataloader batch size')

    # parser.add_argument(
    #     '--input_homography', type=str, default='assets/coco_test_images_homo.txt',
    #     help='Path to the list of image pairs and corresponding homographies')
    # parser.add_argument(
    #     '--input_dir', type=str, default='assets/coco_test_images/',
    #     help='Path to the directory that contains the images')
    parser.add_argument(
        '--results_base_path', type=str, default='output/eval',
        help='base path directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    
    parser.add_argument(
        '--apply_aug', action='store_true',
        help='apply augmentations')

    parser.add_argument(
        '--superglue', default='coco_homo',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--min_matches', type=int, default=12,
        help="Minimum matches required for considering matching")
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', type=str2bool, default=True,
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')

    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # with open(opt.input_homography, 'r') as f:
    #     homo_pairs = f.readlines()

    # if opt.max_length > -1:
    #     homo_pairs = homo_pairs[0:np.min([len(homo_pairs), opt.max_length])]

    # if opt.shuffle:
    #     random.Random(0).shuffle(homo_pairs)
    # download_base_files()
    # download_test_images()
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    try:
        curr_weights_path = str(weights_mapping[opt.superglue])
    except:
        if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
            curr_weights_path = str(opt.superglue)
        else:
            raise ValueError("Given --superglue path doesn't exist or invalid")
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights_path': curr_weights_path,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    opt.img_path = os.path.expanduser(opt.img_path)
    input_dir = Path(opt.img_path)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    
    # create a new directory when we train
    output_dir = Path(opt.results_base_path) / Path(datetime.now().strftime('%Y-%m-%d__%H-%M' + f"_superglue"))

    if output_dir.is_dir():
        print(f"[red]results_path already exists! {output_dir}")
        sys.exit()
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(filename= str(output_dir / "results.log"), format='%(message)s', level=logging.DEBUG)

    def print_and_log(msg):
        print(msg)
        logging.info(msg)

    print_and_log(opt)

    print_and_log('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print_and_log('Will write evaluation results to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print_and_log('Will write visualization images to directory \"{}\"'.format(output_dir))
    
    timer = AverageTimer(newline=True)

    #! INSTEAD OF LOOPING OVER THESE HOMO_PAIRS, WE SHOULD USE THE DATALOADER.
    seen_classes, unseen_classes = random_seen_unseen_class_split(opt.img_path)

    print_and_log(f"seen_classes: {seen_classes}")
    print_and_log(f"unseen_classes: {unseen_classes}")

    dl = DataLoader(opt.img_path,
                                batch_size=opt.batch_size,
                                num_workers=8,
                                shuffle=True,
                                seen_classes=seen_classes,
                                unseen_classes=unseen_classes,
                                use_dataset_rotations=True,
                                template_imgs_only=False, #! THIS WILL LIMIT TO TEMPLATE IMAGES ONLY. AND THEY WILL BE IN TRAIN AND IN VAL/TEST SETS
                                train_transform="don't normalise",
                                val_transform="don't normalise"
                                ) # important flag.



    for dataset_name in ["seen_val", "seen_test", "unseen_test", "test", "all"]:
        errors_dlt = []
        errors_ransac = []
        precisions = []
        recalls = []
        angles_diff_abs = []
        matching_scores = []
        
        print_and_log("\n============================================")
        print_and_log(f"evaluating: {dataset_name}")
        print_and_log("============================================")

        count = 0
        dataset_len = len(dl.dataloaders[dataset_name].dataset)
        print_and_log(f"dataset len: {dataset_len}")

        for batch_i, batch in enumerate(dl.dataloaders[dataset_name]):
            image0s, labels, paths, _, image1s, homo_matrixes, angles = batch

            # iterate over the batch
            for image0, label, path, image1, homo_matrix, angle_gt in zip(image0s, labels, paths, image1s, homo_matrixes, angles):
                count += 1
            # for i, info in enumerate(homo_pairs):

                # print("len(batch_item)", len(batch_item))
                # image0, label, path, _, image1, homo_matrix, angle = batch_item
                # inp0 = image0
                # inp1 = image1

                #! THESE IMAGES ARE NORMALISED. WHICH IS MAYBE NOT WHAT SUPERGLUE EXPECTS!!!

                inp0 = exp_utils.torch_to_grayscale_np_img(image0).astype(np.float32)
                inp1 = exp_utils.torch_to_grayscale_np_img(image1).astype(np.float32)

                #! now do frame2tensor

                inp0 = frame2tensor(inp0, device)
                inp1 = frame2tensor(inp1, device)
                homo_matrix = homo_matrix.float().to(device)

                # print_and_log(f"path {path}")
        
                # split_info = info.strip().split(' ')
                # image_name = split_info[0]
                # homo_info = list(map(lambda x: float(x), split_info[1:]))
                # homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
                stem0 = Path(path).stem
                matches_path = output_dir / '{}_matches.npz'.format(stem0)
                eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
                viz_path = output_dir / '{}_matches.{}'.format(stem0, opt.viz_extension)
                viz_eval_path = output_dir / \
                    '{}_evaluation.{}'.format(stem0, opt.viz_extension)

                # Handle --cache logic.
                do_match = True
                do_eval = opt.eval
                do_viz = opt.viz
                do_viz_eval = opt.eval and opt.viz

                if not (do_match or do_eval or do_viz or do_viz_eval):
                    timer.print('Finished pair {:5} of {:5}'.format(count, dataset_len))
                    continue

                # image0, image1, inp0, inp1, scales0, homo_matrix = read_image_with_homography(input_dir / image_name, homo_matrix, device,
                #                                         opt.resize, 0, opt.resize_float, apply_aug=opt.apply_aug, aug_func=aug_func)

                # if image0 is None or image1 is None:
                #     print('Problem reading image pair: {}'.format(
                #         input_dir/ image_name))
                #     exit(1)
                timer.update('load_image')

                if do_match:
                    # Perform the matching.
                    pred = matching({'image0': inp0, 'image1': inp1})
                    kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
                    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                    matches, conf = pred['matches0'], pred['matching_scores0']
                    timer.update('matcher')

                    # Write the matches to disk.
                    # out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                    #             'matches': matches, 'match_confidence': conf}
                    # np.savez(str(matches_path), **out_matches)

                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                # ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, torch.from_numpy(homo_matrix).to(kp0_torch.device), dist_thresh=3, n_iters=3)
                ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, homo_matrix, dist_thresh=3, n_iters=3)
                ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
                gt_match_vec = np.ones((len(matches), ), dtype=np.int32) * -1
                gt_match_vec[ma_0] = ma_1
                corner_points = np.array([[0,0], [0, image0.shape[0]], [image0.shape[1], image0.shape[0]], [image0.shape[1], 0]]).astype(np.float32)
                if do_eval:
                    if len(mconf) < opt.min_matches:
                        out_eval = {'error_dlt': -1,
                                    'error_ransac': -1,
                                    'precision': -1,
                                    'recall': -1
                                    }
                        #non matched points will not be considered for evaluation
                        # np.savez(str(eval_path), **out_eval)
                        timer.update('eval')
                        print('Skipping {} due to inefficient matches'.format(count))
                        continue
                    sort_index = np.argsort(mconf)[::-1][0:4]
                    est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
                    est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)

                    # ! Seb's code
                    # get rotation from homography
                    
                    def angle_from_homo(homo):
                        # https://stackoverflow.com/questions/58538984/how-to-get-the-rotation-angle-from-findhomography
                        u, _, vh = np.linalg.svd(homo[0:2, 0:2])
                        R = u @ vh
                        angle = math.atan2(R[1,0], R[0,0]) # angle between [-pi, pi)
                        return angle
                    
                    def difference_angle(angle1, angle2):
                        difference = (angle1 - angle2) % (2*np.pi)
                        difference = np.where(difference > np.pi, difference - 2*np.pi, difference)
                        return difference

                    homo_matrix_cpu = homo_matrix.cpu().numpy()
                    
                    angle_est = angle_from_homo(est_homo_ransac)
                    angle_gt_from_homo = angle_from_homo(homo_matrix_cpu)
                    angle_diff_abs = np.abs(difference_angle(angle_est, angle_gt_from_homo))
                    # print("angle_est", angle_est, "angle_gt", angle_gt)
                    print(f"difference {np.round(angle_diff_abs, 4)}")

                    # ! end: Seb's code


                    corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
                    corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(1)
                    corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), homo_matrix_cpu).squeeze(1)
                    error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
                    error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
                    match_flag = (matches[ma_0] == ma_1)
                    precision = match_flag.sum() / valid.sum()
                    fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
                    recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
                    # Write the evaluation results to disk.
                    out_eval = {'error_dlt': error_dlt,
                                'error_ransac': error_ransac,
                                'precision': precision,
                                'recall': recall,
                                'angle': angle_diff_abs,
                                }
                    # np.savez(str(eval_path), **out_eval)

                    errors_dlt.append(error_dlt)
                    errors_ransac.append(error_ransac)
                    precisions.append(precision)
                    recalls.append(recall)
                    angles_diff_abs.append(angle_diff_abs)

                    timer.update('eval')

                if do_viz:
                    # Visualize the matches.
                    color = cm.jet(mconf)
                    text = [
                        'SuperGlue',
                        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                        'Matches: {}'.format(len(mkpts0)),
                    ]

                    # Display extra parameter info.
                    k_thresh = matching.superpoint.config['keypoint_threshold']
                    m_thresh = matching.superglue.config['match_threshold']
                    small_text = [
                        'Keypoint Threshold: {:.4f}'.format(k_thresh),
                        'Match Threshold: {:.2f}'.format(m_thresh),
                        'Image Pair: {}:{}'.format(stem0, stem0),
                    ]

                    make_matching_plot(
                        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                        text, viz_path, opt.show_keypoints,
                        opt.fast_viz, opt.opencv_display, 'Matches', small_text)

                    timer.update('viz_match')

                timer.print('Finished pair {:5} of {:5}'.format(count, dataset_len))

        if opt.eval:
        # Collate the results into a final table and print to terminal.



            # for info in homo_pairs:
            #     split_info = info.strip().split(' ')
            #     image_name = split_info[0]
            #     stem0 = Path(image_name).stem
            #     eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
            #     results = np.load(eval_path)
            #     if results['precision'] == -1:
            #         continue

            
            thresholds = [5, 10, 25]
            aucs_dlt = pose_auc(errors_dlt, thresholds)
            aucs_ransac = pose_auc(errors_ransac, thresholds)
            aucs_dlt = [100.*yy for yy in aucs_dlt]
            aucs_ransac = [100.*yy for yy in aucs_ransac]
            prec = 100.*np.mean(precisions)
            rec = 100.*np.mean(recalls)
            
            angle_mean = np.mean(angles_diff_abs)
            angle_std = np.std(angles_diff_abs)
            angle_median = np.median(angles_diff_abs)

            angles_diff_square = np.square(angles_diff_abs)
            angle_square_mean = np.mean(angles_diff_square)
            angle_square_std = np.std(angles_diff_square)
            angle_square_median = np.median(angles_diff_square)

            print_and_log('Evaluation Results (mean over {} pairs):'.format(dataset_len))
            print_and_log("For DLT results...")
            print_and_log('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
            print_and_log('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
                aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec))
            print_and_log("For homography results...")
            print_and_log('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
            print_and_log('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
                aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec))
            print_and_log("angles abs:")
            print_and_log("med\t mean\t std\t")
            print_and_log('{:.3}\t {:.3}\t {:.3}\t'.format(angle_median, angle_mean, angle_std))

            print_and_log("angles square:")
            print_and_log("med\t mean\t std\t")
            print_and_log('{:.3}\t {:.3}\t {:.3}\t'.format(angle_square_median, angle_square_mean, angle_square_std))




