import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import math
import json

def overlap_distance_calculate(mat_true, mat_pred, min_valid_range=10, parallel_workers=1):
    """
    mat_true, mat_pred: 2d matrices, with 0s and 1s only
    min_valid_range: the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    calculate_distance: when True this will not only calculate overlapping pixels
        but also the distances between nearesttrue and predicted pixels
    """

    lowest_dist_pairs=[]
    points_done_pred=set()
    points_done_true=set()

    # first calculate the overlapping pixels
    mat_overlap=mat_pred*mat_true
    for x_true, y_true in tqdm(np.argwhere(mat_overlap==1)):
        lowest_dist_pairs.append((((x_true, y_true), (x_true, y_true)), 0.0))
        points_done_true.add((x_true, y_true))
        points_done_pred.add((y_true, x_true))
    print('len(lowest_dist_pairs) by overlapping only:', len(lowest_dist_pairs))

    diagonal_length=math.sqrt(math.pow(mat_true.shape[0], 2)+ math.pow(mat_true.shape[1], 2))
    min_valid_range=int((min_valid_range*diagonal_length)/100) # in pixels
    print('calculated pixel min_valid_range:', min_valid_range)

    def nearest_pixels(x_true, y_true):
        result=[]
        # find all the points in pred withing min_valid_range rectangle
        mat_pred_inrange=mat_pred[
         max(x_true-min_valid_range, 0): min(x_true+min_valid_range, mat_true.shape[1]),
            max(y_true-min_valid_range, 0): min(y_true+min_valid_range, mat_true.shape[0])
        ]
        for x_pred_shift, y_pred_shift in np.argwhere(mat_pred_inrange==1):
            y_pred=max(y_true-min_valid_range, 0)+y_pred_shift
            x_pred=max(x_true-min_valid_range, 0)+x_pred_shift
            if (x_pred, y_pred) in points_done_pred:
                continue
            # calculate eucledean distances
            dist_square=math.pow(x_true-x_pred, 2)+math.pow(y_true-y_pred, 2)
            result.append((((x_true, y_true), (x_pred, y_pred)), dist_square))
        return result

    candidates=[(x_true, y_true) for x_true, y_true in tqdm(np.argwhere(mat_true==1)) if (x_true, y_true) not in points_done_true]
    distances=Parallel(n_jobs=parallel_workers)(delayed(nearest_pixels)(x_true, y_true) for x_true, y_true in tqdm(candidates))
    distances = [item for sublist in distances for item in sublist]

    # sort based on distances
    distances=sorted(distances, key=lambda x: x[1])

    # find the lowest distance pairs
    for ((x_true, y_true), (x_pred, y_pred)), distance in tqdm(distances):
        if ((x_true, y_true) in points_done_true) or ((x_pred, y_pred) in points_done_pred):
            # do not consider a taken point again
            continue
        # normalize all distances by diving by the diagonal length
        lowest_dist_pairs.append((((x_true, y_true), (x_pred, y_pred)), math.sqrt(float(distance))/diagonal_length))
        points_done_true.add((x_true, y_true))
        points_done_pred.add((x_pred, y_pred))

    return lowest_dist_pairs

def detect_difficult_pixels(map_image, binary_raster, legend_coor, plot=True, set_false_as='hard'):
    """
    map_image: the image array for the map image
    binary_raster: 2D array of any channel (out of 3 present) from the true binary raster image
    legend_coor: coordinate for the legend feature, from the legend json file
    plot: plots different rasters
    set_false_as: when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """

    plt.rcParams["figure.figsize"] = (15,22)

    # detect pixels based on color of legend
    if legend_coor is not None:
        pred_by_color=match_by_color(map_image.copy(), legend_coor, color_range=20)
        if plot:
            print('predicted based on color of legend:')
            plt.imshow(pred_by_color)
            plt.show()
    else:
        return

    pred_by_color=(1-pred_by_color).astype(int) # flip, so the unpredicted become hard pixels
    pred_by_color=binary_raster*pred_by_color # keep only the part within the true polygon
    if plot:
        print('hard pixel (flipped predictions) using the color predictions (in the polygon range):')
        plt.imshow(pred_by_color)
        plt.show()

    if set_false_as=='hard':
        # the pixels that are not within the true polygon should are deemed hard pixels
        final_hard_pixels=(1-binary_raster)|pred_by_color
    else:
        # the outside pixels will be deemed easy!
        final_hard_pixels=pred_by_color

    if plot:
        print('final hard pixels (merged):')
        plt.imshow(final_hard_pixels)
        plt.show()

    return final_hard_pixels

def match_by_color(img, legend_coor, color_range=20):
    """
    img: the image array for the map image
    legend_coor: coordinate for the legend feature, from the legend json file
    """
    # get the legend coors and the predominant color
    (x_min, y_min), (x_max, y_max) = legend_coor
    legend_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    # take the median of the colors to find the predominant color
    r=int(np.median(legend_img[:,:,0]))
    g=int(np.median(legend_img[:,:,1]))
    b=int(np.median(legend_img[:,:,2]))
    sought_color=[r, g, b]
    print('matching the color:', sought_color, 'with color range:', color_range)
    # capture the variations of legend color due to scanning errors
    lower = np.array([x - color_range for x in sought_color], dtype="uint8")
    upper = np.array([x + color_range for x in sought_color], dtype="uint8")
    # create a mask to only preserve current legend color in the basemap
    mask = cv2.inRange(img, lower, upper)
    detected = cv2.bitwise_and(img, img, mask=mask)
    # convert to grayscale
    detected_gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.threshold(detected_gray, 127, 255, cv2.THRESH_BINARY)[1]
    # convert the grayscale image to binary image
    pred_by_color = img_bw.astype(float) / 255
    return pred_by_color

def feature_f_score(map_image_path, predicted_raster_path, true_raster_path, legend_json_path=None, min_valid_range=.25,
                      difficult_weight=.7, set_false_as='hard', plot=True):

    """
    map_image_path: path to the the actual map image
    predicted_raster_path: path to the the predicted binary raster image
    true_raster_path: path to the the true binary raster image
    legend_json_path: (only used for polygons) path to the json containing the coordinates for the corresponding legend (polygon) feature
    min_valid_range: (only used for points and lines) the maximum distance in % of the largest length in the image i.e. the diagonal
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    difficult_weight: (only used for polygons) float within [0, 1], weight for the difficlut pixels in the scores (only for polygins)
    set_false_as: (only used for polygons) when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """

    img=cv2.imread(map_image_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    true_raster=cv2.imread(true_raster_path)
    true_raster=true_raster[:,:,0]

    predicted_raster=cv2.imread(predicted_raster_path)
    predicted_raster=predicted_raster[:,:,0]

    # plot: overlay the true and predicted values on the map image
    if plot:
        im_copy=img.copy()
        for center in np.argwhere(predicted_raster==1):
            cv2.circle(im_copy, (center[1], center[0]), 1, (0,255,0), -1) # green
        print('Predicted raster overlayed on map image:')
        plt.rcParams["figure.figsize"] = (15,22)
        plt.imshow(im_copy)
        plt.show()
        im_copy=img.copy()
        for center in np.argwhere(true_raster==1):
            cv2.circle(im_copy, (center[1], center[0]), 1, (255,0,0), -1) # red
        print('True raster overlayed on map image:')
        plt.imshow(im_copy)
        plt.show()


    legend_feature=os.path.basename(true_raster_path).replace(os.path.basename(map_image_path).replace('.tif', '')+'_', '').replace('.tif', '')
    feature_type=legend_feature.split('_')[-1]
    print('feature type:', feature_type)

    legend_coor=None
    if legend_json_path is not None:
        legends=json.loads(open(legend_json_path, 'r').read())
        for shape in legends['shapes']:
            if legend_feature ==shape['label']:
                legend_coor=legends['shapes'][0]['points']
        print('legend_coor:', legend_coor)

    mat_true, mat_pred=true_raster, predicted_raster

    if feature_type in ['line', 'pt']: # for point and lines
        lowest_dist_pairs=overlap_distance_calculate(mat_true, mat_pred,
                                                     min_valid_range=min_valid_range)
        print('len(lowest_dist_pairs):', len(lowest_dist_pairs))
        sum_of_similarities=sum([1-item[1] for item in lowest_dist_pairs])
        print('sum_of_similarities:', sum_of_similarities)
        print('num all pixel pred:', len(np.argwhere(mat_pred==1)))
        print('num all pixel true:', len(np.argwhere(mat_true==1)))
        precision=sum_of_similarities/len(np.argwhere(mat_pred==1))
        recall=sum_of_similarities/len(np.argwhere(mat_true==1))
    else: # for polygon

        overlap=mat_true*mat_pred
        num_overlap=len(np.argwhere(overlap==1))
        print('num_overlap:', num_overlap)
        num_mat_pred=len(np.argwhere(mat_pred==1))
        print('num_mat_pred:', num_mat_pred)
        num_mat_true=len(np.argwhere(mat_true==1))
        print('num_mat_true:', num_mat_true)

        if difficult_weight is not None:

            difficult_pixels=detect_difficult_pixels(img, true_raster, legend_coor=legend_coor, set_false_as=set_false_as, plot=plot)

            num_overlap_difficult=len(np.argwhere((overlap*difficult_pixels)==1))
            print('num_overlap_difficult:', num_overlap_difficult)
            num_overlap_easy=len(np.argwhere((overlap-(overlap*difficult_pixels))==1))
            print('num_overlap_easy:', num_overlap_easy)
            points_from_overlap=(num_overlap_difficult*difficult_weight)+(num_overlap_easy*(1-difficult_weight))
            print('points_from_overlap:', points_from_overlap)

            num_mat_pred_difficult=len(np.argwhere((mat_pred*difficult_pixels)==1))
            print('num_mat_pred_difficult:', num_mat_pred_difficult)
            num_mat_pred_easy=len(np.argwhere((mat_pred-(mat_pred*difficult_pixels))==1))
            print('num_mat_pred_easy:', num_mat_pred_easy)
            total_pred=(num_mat_pred_difficult*difficult_weight)+(num_mat_pred_easy*(1-difficult_weight))
            print('total prediction points contended:', total_pred)
            precision=points_from_overlap/total_pred


            num_mat_true_difficult=len(np.argwhere((mat_true*difficult_pixels)==1))
            print('num_mat_true_difficult:', num_mat_true_difficult)
            num_mat_true_easy=len(np.argwhere((mat_true-(mat_true*difficult_pixels))==1))
            print('num_mat_true_easy:', num_mat_true_easy)
            total_true=(num_mat_true_difficult*difficult_weight)+(num_mat_true_easy*(1-difficult_weight))
            print('total true points to be had:', total_true)
            recall=points_from_overlap/total_true

        else:
            precision=num_overlap/num_mat_pred
            recall=num_overlap/num_mat_true


    # calculate f-score
    if precision+recall!=0:
        f_score=(2 * precision * recall) / (precision + recall)
    else:
        f_score=0

    return precision, recall, f_score
