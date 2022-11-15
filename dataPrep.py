# Data Prep Functions
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from math import floor, ceil
from patchify import patchify

def crop_map(image):
    ### image is a np array of the base image you want to crop the map out of. This function returns the same size image with all of the non-map content masked to black.
    ### If we need to explain this this function basically just does contour dectection and selects the largest feature in the image, this is almost always the map. there are a couple extra steps to deal with black borders and noise in the image as well
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # greyscale image
    # Detect Background Color
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill borders to deal black borders
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Keeping only the largest detected contour.
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Mask everything that is not the map
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, pts=[contour], color=(255,255,255))
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def patch_map_directory(sourceDir, outputDir, patchDims, patchOverlap, discardThreshold=0.8, discardLabelPatchs=False):
    total_patchs = 0
    # Iterate through each map's json file.
    pbar = tqdm(os.listdir(sourceDir))
    for file in pbar:
        if not file.endswith(".json"):
            continue
        
        # Load Json
        with open(os.path.join(sourceDir, file)) as fh:
            json_data = json.load(fh)

        # Load Base Image
        json_data['imagePath'] = file.split('.')[0] + '.tif'

        base_filePath = os.path.join(sourceDir, json_data['imagePath'])
        base_img = cv2.imread(base_filePath)
        if base_img is None:
            print('Could not find base image {}, Skipping'.format(base_filePath))
            continue

        # Mask Base Image
        masked_img = crop_map(base_img)

        # Patch Base Image
        masked_patchs = patchify(masked_img, patchDims, step=(patchDims[0]-patchOverlap))

        # Discard non-map patches
        discard_patchs = []
        discard_threshold = discardThreshold * (patchDims[0] * patchDims[1] * patchDims[2])
        # Discard patches that are 'DISCARD_MASK_THRESHOLD'% part of the masked out area.
        for row in range(0,masked_patchs.shape[0]):
            for col in range(0,masked_patchs.shape[1]):
                if np.count_nonzero(masked_patchs[row][col]) < discard_threshold:
                    discard_patchs.append('{}_{}'.format(row, col))
        
        empty_patch_count = len(discard_patchs)

        # Discard patches with annotations in them
        if discardLabelPatchs:
            for shape in json_data['shapes']:
                if shape['label'].split('_')[-1] == 'pt':
                    feature_filePath = os.path.join(sourceDir, json_data['imagePath'].split('.')[0] + '_' + shape['label'] + '.tif')
                    feature_img = cv2.imread(feature_filePath, cv2.IMREAD_GRAYSCALE)
                    if feature_img is None:
                        print('Could not find feature image {}, Skipping'.format(feature_filePath))
                        continue

                    # Patch the feature label mask
                    feature_patchs = patchify(feature_img, (patchDims[0], patchDims[1]), step=(patchDims[0]-patchOverlap))

                    # Iterate over each patch of the mask
                    for row in range(0,feature_patchs.shape[0]):
                        for col in range(0,feature_patchs.shape[1]):
                            if '{}_{}'.format(row, col) not in discard_patchs and np.count_nonzero(feature_patchs[row][col]) > 0:
                                discard_patchs.append('{}_{}'.format(row, col))
        
        # Save Patches
        saved_patchs = 0
        for row in range(0,masked_patchs.shape[0]):
            for col in range(0,masked_patchs.shape[1]):
                pid = '{}_{}'.format(row, col)
                if pid not in discard_patchs:
                    # Save patch image
                    patch_image_fileName = '{}_{}.tif'.format(json_data['imagePath'].split('.')[0], pid)
                    patch_image_filePath = os.path.join(outputDir, patch_image_fileName)
                    cv2.imwrite(patch_image_filePath, masked_patchs[row][col].squeeze())
                    saved_patchs += 1
        
        total_patchs += saved_patchs
        pbar.set_description('Total Patches Created : {}'.format(total_patchs))
    #print('Created {} patches, Success!'.format(total_patchs))

def build_legend_dictionary(sourceDir):
    pt_classes = {}
    # Iterate through each json file.
    pbar = tqdm(os.listdir(sourceDir))
    for file in pbar:
        if not file.endswith(".json"):
            continue
        
        # Load Json
        with open(os.path.join(sourceDir, file)) as fh:
            json_data = json.load(fh)

        for shape in json_data['shapes']:
            if shape['label'].split('_')[-1] == 'pt':
                if shape['label'] not in pt_classes:
                    # Load Base Image
                    json_data['imagePath'] = file.split('.')[0] + '.tif'
                    base_filePath = os.path.join(sourceDir, json_data['imagePath'])
                    base_img = cv2.imread(base_filePath)

                    x1 = floor(shape['points'][0][0])
                    y1 = floor(shape['points'][0][1])
                    x2 = floor(shape['points'][1][0])
                    y2 = floor(shape['points'][1][1])

                    pt_classes[shape['label']] = base_img[y1:y2, x1:x2]

        pbar.set_description('Total Classes Found : {}'.format(len(pt_classes.keys())))
    print('There are {} classes in the dataset'.format(len(pt_classes.keys())))
    return pt_classes

# Utility math function
def clamp(value, lower_bound=0.0, upper_bound=1.0):
    return max(min(value, upper_bound), lower_bound)

def place_glyph(img, pat, x, y, thresh=150):
    """                                                                                                                                                    
    img: image to be modified                                                                                                                              
    pat: patch to place on the img                                                                                                                         
    x,y: pixel coordinates to place patch, upper left corner                                                                                               
    """
    pat_gr = cv2.cvtColor(pat, cv2.COLOR_BGR2GRAY)
    pat_mask = pat_gr < thresh
    sh = pat.shape
    img[x:x+sh[0],y:y+sh[1]:,:][pat_mask,:] = pat[pat_mask,:]
    return img

def create_synthetic_training_data(patchDir, outputDir, legendDir, patchCount, glyphDistribution='Uniform', maxGlyphs=12, minGlyphs=0, randomRotation=False):
    # Create output directories
    if not os.path.exists(os.path.join(outputDir, 'images')):
        os.makedirs(os.path.join(outputDir, 'images'))
    if not os.path.exists(os.path.join(outputDir, 'labels')):
        os.makedirs(os.path.join(outputDir, 'labels'))

    distribution_options = ['uniform', 'stddev']
    bgpatchs = os.listdir(patchDir)
    np.random.shuffle(bgpatchs)

    # Load legend labels
    labels = {}
    for label in os.listdir(legendDir):
        labels[label] = cv2.imread(os.path.join(legendDir, label))
    label_keys = list(labels.keys())
    
    # Save human readable label index
    idx = ""
    i = 0
    for key in label_keys:
        idx += '{} {}\n'.format(i, key)
        i += 1
    with open(os.path.join(outputDir, 'Class Index.txt'), 'w') as fh:
        fh.write(idx)

    if glyphDistribution.lower() not in distribution_options:
        print('WARNING creating synthethic training data : glyphDistribution not set to a valid option. defaulting to Uniform distribution')
        glyphDistribution = 'uniform'
    
    # Generate Patchs
    for i in tqdm(range(0,patchCount)):
        glyph_count = 0
        if glyphDistribution.lower() == 'uniform':
            glyph_count = np.random.randint(maxGlyphs-minGlyphs) + minGlyphs
        elif glyphDistribution.lower() == 'stddev':
            glyph_count = round(clamp(np.random.normal(0.5, 0.2, 10)) * (maxGlyphs-minGlyphs))
        else:
            print('how the hell did i get here')
        
        # Get background of patch
        patch = cv2.imread(os.path.join(patchDir, bgpatchs[i%len(bgpatchs)]))

        annotations = ""
        for p in np.random.rand(glyph_count, 2):
            class_num = np.random.randint(len(label_keys))
            label = labels[label_keys[class_num]]
            
            annotations += ('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_num, p[0], p[1], label.shape[0]/patch.shape[0], label.shape[1]/patch.shape[1]))
            
            x = clamp(floor(p[0]*patch.shape[0]), 0, patch.shape[0]-label.shape[0])
            y = clamp(floor(p[1]*patch.shape[1]), 0, patch.shape[1]-label.shape[1])
            patch = place_glyph(patch, label, x, y)

        # Save patch and annotations
        patch_filepath = os.path.join(outputDir, 'images', 'synpatch_{}.tif'.format(i))
        cv2.imwrite(patch_filepath, patch)
        label_filepath = os.path.join(outputDir, 'labels', 'synpatch_{}.txt'.format(i))
        with open(label_filepath, 'w') as fh:
            fh.write(annotations)

def main():
    
    sourceDir = 'data/training'
    legendDir = 'data/legends'
    patchDir = 'data/patchs'
    synDataDir = 'data/synthetic'

    patchDims = (512, 512, 3)
    patchOverlap = 64
    contentThreshold = 0.9

    sythPatchsToMake = 1000

    if not os.path.exists(legendDir):
        os.makedirs(legendDir)
    if not os.path.exists(patchDir):
        os.makedirs(patchDir)
    if not os.path.exists(synDataDir):
        os.makedirs(synDataDir)

    print('Running synthetic datagen with:\n \
            Base Dataset : {}\n \
            Output Directory : {}\n \
            Generated Patchs : {}\n \
            Patch Size: {}\n \
            Patch Overlap {}'.format(sourceDir, synDataDir, sythPatchsToMake, patchDims, patchOverlap))

    print('Getting valid background patchs from training set')
    # Generate blank backgrounds for synthethic patchs
    patch_map_directory(sourceDir, patchDir, patchDims, patchOverlap, contentThreshold, True)
    
    print('Creating label Dictionary')
    # Save legends
    pt_legends = build_legend_dictionary(sourceDir)
    for label in pt_legends:
        cv2.imwrite(os.path.join(legendDir, '{}.tif'.format(label)),pt_legends[label])
    
    print('Creating synthetic patchs')
    # Build sythethic dataset from patchs and legends
    create_synthetic_training_data(patchDir, synDataDir, legendDir, sythPatchsToMake)


if __name__ == '__main__':
    main()

