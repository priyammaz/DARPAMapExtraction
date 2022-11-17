import os
import cv2
import json
import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt

# Config
# Data Folders
sourceDataDir = 'data/final_dataset' # Directory containing the orginal maps and masks
predictPatchsDir = 'data/final_annotations'    # Directory containing patched maps and annotations
outputMaskDir = 'data/final_masks'
classIndex = 'data/finalIndex.txt'

# PatchSettings
patch_dims = (512,512)
patch_overlap = 64
patch_step = patch_dims[1]-patch_overlap

if not os.path.exists(outputMaskDir):
    os.makedirs(outputMaskDir)

def str_to_anno(rawtxt, patch_dims):
    c, px, py, pw, ph = [float(x) for x in rawtxt.split(' ')]
    c = int(c)
    mx = floor(px * patch_dims[0])
    my = floor(py * patch_dims[1])
    mw = floor(pw * patch_dims[0])
    mh = floor(ph * patch_dims[1])
    return [c, mx, my, mw, mh]

def convert_patch_to_map(patch_annotation, map_dims, patch_dims, patch_index, overlap=0):
    c, py, px, ph, pw = [float(x) for x in patch_annotation.split(' ')]
    c = int(c)
    mx = (floor(px * patch_dims[0]) + patch_index[0] * (patch_dims[0] - overlap)) / map_dims[0]
    my = (floor(py * patch_dims[1]) + patch_index[1] * (patch_dims[1] - overlap)) / map_dims[1]
    mw = floor(pw * patch_dims[0]) / map_dims[0]
    mh = floor(ph * patch_dims[1]) / map_dims[1]
    return [c, mx, my, mw, mh]

def unpatch_annotations(json_data, base_img):
    # Load annotations
    annotations = []
    for filename in os.listdir(predictPatchsDir):
        if json_data['imagePath'].split('.')[0] in filename:
            with open(os.path.join(predictPatchsDir,filename)) as fh:
                row, col = [int(x) for x in filename.split('.')[0].split('_')[-2:]]
                for line in fh.readlines():
                    annotations.append(convert_patch_to_map(line, base_img.shape, patch_dims, (row, col), overlap=patch_overlap))
    
    if len(annotations) == 0:
        # Mention no predictions were found for map
        print('No annotations were found for map')

    return annotations

def read_class_index(filepath):
    label_dict = {}
    with open(filepath) as fh:
        for line in fh.readlines():
            content = line.split(' ')
            label_dict[content[1].split('.')[0]] = content[0]
    return label_dict

def main():
    label_dict = read_class_index(classIndex)

    # Iterate through each map's json file.
    for file in os.listdir(sourceDataDir):
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
            print('Could not find base image {}, Skipping'.format(base_filePath))
            continue
        
        annotations = unpatch_annotations(json_data, base_img)
        
        for shape in json_data['shapes']:
            if shape['label'].split('_')[-1] == 'pt':
                feature_filename = '{}_{}.tif'.format(json_data['imagePath'].split('.')[0], shape['label'])
                print('Creating feature mask {}'.format(feature_filename))

                if shape['label'] in label_dict:
                    class_id = label_dict[shape['label']][0]
                else:
                    print(shape['label'] + ' was not present in class ids')
                    continue
                feature_annotations = [row for row in annotations if row[0] == class_id]
                if len(feature_annotations) != 0: 
                    feature_map = np.zeros_like(base_img)
                    for row in feature_annotations:
                        x = floor(row[1] * feature_map.shape[0])
                        y = floor(row[2] * feature_map.shape[1])
                        feature_map[x][y][:] = 1
                    cv2.imwrite(os.path.join(outputMaskDir, feature_filename), feature_map.squeeze())
                else :
                    print('No predictions for {}'.format(shape['label']))

if __name__ == '__main__':
    main()