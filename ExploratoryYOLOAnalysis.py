import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from math import floor, ceil
from matplotlib import pyplot as plt
from patchify import patchify

# Config
logFile = 'patching.log'
# Data Folders
sourceDataDir = 'smallTrain' # Directory containing the orginal maps and masks
yoloDataFolder = 'smallData'    # Directory containing patched maps and annotations

# PatchSettings
patch_dims = (512,512)
patch_overlap = 64
patch_step = patch_dims[1]-patch_overlap

# These paths are expected by yolo probably shouldn't change them
patchTrainFolder = 'images/train'
patchLabelFolder = 'labels/train'

# Utility math function
def clamp(value, lower_bound=0.0, upper_bound=1.0):
    return max(min(value, upper_bound), lower_bound)

# Generates yolo formatted annotations from a point mask image with the size of the bounding box given by bbox_shape
def generate_yolo_annotations(points_mask, bbox_shape, class_id):
    # Get list of non-zero pixels 
    points = np.argwhere(points_mask)
    # Convert point to yolo format (class_id, x_center, y_center, width, height)
    annotations = []
    for point in points:
        x = clamp(point[1] / points_mask.shape[0])
        y = clamp(point[0] / points_mask.shape[1])
        w = clamp(bbox_shape[0] / points_mask.shape[0])
        h = clamp(bbox_shape[1] / points_mask.shape[1])
        
        annotations.append([class_id,x,y,w,h])
    return annotations

# Utility function to plot list of annotations on image
def plot_bounding_box(image, annotations):
    #colors = [crimson, deepskyblue, forestgreen, gold, mediumOrchid, papayawhip, sienna, silver, greenyellow, firebrick, navy]
    colors = [(220, 20, 60),(0, 191, 255),(34, 139, 34),(255, 215, 0),(186, 85, 211),(255, 239, 213),(160, 82, 45),(192, 192, 192),(173, 255, 47),(178, 34, 34),(0, 0, 128)]
    annotated_img = image.copy()
    w, h = annotated_img.shape[0:2]
    for bb in annotations:
        l = int((bb[1] - bb[3] / 2) * w)
        r = int((bb[1] + bb[3] / 2) * w)
        t = int((bb[2] - bb[4] / 2) * h)
        b = int((bb[2] + bb[4] / 2) * h)
        
        if l < 0:
            l = 0
        if r > w - 1:
            r = w - 1
        if t < 0:
            t = 0
        if b > h - 1:
            b = h - 1

        cv2.rectangle(annotated_img, (l,t), (r,b), colors[bb[0]%len(colors)], 3)
    return annotated_img



def yolo_data_prep():
    print('Starting Data Prep')
    log_fh = open(logFile, 'w')
    print('Starting Data Prep with cfg:\n \
            \tSource Dir : {}\n \
            \tOutput Dir : {}\n \
            \tPatch Size : ({},{})\n \
            \tPatch Overlap : {}'.format(sourceDataDir, yoloDataFolder, patch_dims[0], patch_dims[1], patch_overlap), file=log_fh)

    total_patchs = 0
    pt_classes = {}
    class_id = 0

    if not os.path.exists(os.path.join(yoloDataFolder, patchTrainFolder)):
        os.makedirs(os.path.join(yoloDataFolder, patchTrainFolder))
    if not os.path.exists(os.path.join(yoloDataFolder, patchLabelFolder)):
        os.makedirs(os.path.join(yoloDataFolder, patchLabelFolder))
    
    print('Starting map patching', file=log_fh)
    # Iterate through each map's json file.
    pbar = tqdm(os.listdir(sourceDataDir))
    for file in pbar:
        if not file.endswith(".json"):
            continue
        
        # Load Json
        with open(os.path.join(sourceDataDir, file)) as fh:
            json_data = json.load(fh)

        # Load Base Image
        json_data['imagePath'] = file.split('.')[0] + '.tif'

        base_filePath = os.path.join(sourceDataDir, json_data['imagePath'])
        base_img = cv2.imread(base_filePath)
        if base_img is None:
            continue
        base_patchs = patchify(base_img, (*patch_dims,3), step=patch_step)
        
        # Save base patches to patch training data
        print('{} : {} x {} = {} Patches'.format(json_data['imagePath'], base_patchs.shape[0], base_patchs.shape[1], base_patchs.shape[0]*base_patchs.shape[1]), file=log_fh)
        #for row in range(0,base_patchs.shape[0]):
        #    for col in range(0,base_patchs.shape[1]):
        #        patch_fileName = '{}_{}_{}.tif'.format(json_data['imagePath'].split('.')[0],row, col)
        #        patch_filePath = os.path.join(yoloDataFolder, patchTrainFolder, patch_fileName)
        #        cv2.imwrite(patch_filePath, base_patchs[row][col].squeeze())
        
        total_patchs += base_patchs.shape[0]*base_patchs.shape[1]
        pbar.set_description('Total Patches Created : {}'.format(total_patchs))

        annotations = {}
        # Iterate over each point mask in the map
        for shape in json_data['shapes']:
            if shape['label'].split('_')[-1] == 'pt':
                #print(json_data['imagePath'].split('.')[0] + '_' + shape['label'], file=log_fh)
                # Load the feature label mask
                feature_filePath = os.path.join(sourceDataDir, json_data['imagePath'].split('.')[0] + '_' + shape['label'] + '.tif')
                feature_img = cv2.imread(feature_filePath, cv2.IMREAD_GRAYSCALE)
                
                # Keep track of class ids and track stats.
                if shape['label'] in pt_classes:
                    pt_classes[shape['label']][1] += 1
                    pt_classes[shape['label']][2] += np.count_nonzero(feature_img)
                else:
                    pt_classes[shape['label']] = [class_id, 1, np.count_nonzero(feature_img)]
                    class_id += 1

                # Patch the feature label mask
                feature_patchs = patchify(feature_img, patch_dims, step=patch_step)

                # Legend size
                lgd_width = ceil(abs(shape['points'][1][0] - shape['points'][0][0]))
                lgd_height = ceil(abs(shape['points'][1][1] - shape['points'][0][1]))
                
                # Iterate over each patch of the mask
                for row in range(0,feature_patchs.shape[0]):
                    for col in range(0,feature_patchs.shape[1]):
                        if np.argwhere(feature_patchs[row][col]).any():
                            pid = '{}_{}'.format(row,col)
                            if pid not in annotations:
                                annotations[pid] = generate_yolo_annotations(feature_patchs[row][col], (lgd_width,lgd_height), pt_classes[shape['label']][0])
                            else:
                                annotations[pid].extend(generate_yolo_annotations(feature_patchs[row][col], (lgd_width,lgd_height), pt_classes[shape['label']][0]))
        
        # Save the annotations
        for patch in annotations:
            patch_annotation_fileName = '{}_{}.txt'.format(json_data['imagePath'].split('.')[0], patch)
            patch_annotation_filePath = os.path.join(yoloDataFolder, patchLabelFolder, patch_annotation_fileName)
            
            annotation_str = ''
            for point in annotations[patch]:
                annotation_str += '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(point[0], point[1], point[2], point[3], point[4])
            annotation_str = annotation_str.strip()

            # Save patch annotation
            with open(patch_annotation_filePath, 'w') as fh:
                fh.write(annotation_str)

            # Save patch image
            patch_image_fileName = '{}_{}.tif'.format(json_data['imagePath'].split('.')[0], patch)
            patch_image_filePath = os.path.join(yoloDataFolder, patchTrainFolder, patch_image_fileName)
            cv2.imwrite(patch_image_filePath, base_patchs[row][col].squeeze())

    print('Generated {} patches'.format(total_patchs))
    print('There are {} classes in the dataset'.format(len(pt_classes.keys())))
    print('class_name : [id, number of maps it occurs in]')
    print(pt_classes)

    print('Generated {} patches'.format(total_patchs), file=log_fh)
    print('There are {} classes in the dataset'.format(len(pt_classes.keys())), file=log_fh)
    print('class_name : [id, number of maps it occurs in]', file=log_fh)
    print(pt_classes, file=log_fh)

def clean_empty_patchs():
    matches = 0
    for file in os.listdir(os.path.join(yoloDataFolder, patchTrainFolder)):
        if not os.path.exists(os.path.join(yoloDataFolder, patchLabelFolder, file.split('.')[0] + '.txt')):
            os.remove(os.path.join(yoloDataFolder, patchTrainFolder, file))
            matches += 1
        
    print('Deleted {} image patches with no annotation data'.format(matches))

def main():
    yolo_data_prep()
    #clean_empty_patchs()

if __name__ == '__main__':
    main()