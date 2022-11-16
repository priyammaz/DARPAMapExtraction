import os
import cv2
import json
import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt

# Config
# Data Folders
sourceDataDir = 'validation' # Directory containing the orginal maps and masks
predictPatchsDir = 'val_predict'    # Directory containing patched maps and annotations
outputMaskDir = 'val_masks'
yoloClassDictionary = 'yoloClasses.csv'

# PatchSettings
patch_dims = (512,512)
patch_overlap = 64
patch_step = patch_dims[1]-patch_overlap

if not os.path.exists(outputMaskDir):
    os.makedirs(outputMaskDir)

def convert_patch_to_map(patch_annotation, map_dims, patch_dims, patch_index, overlap=0):
    c, px, py, pw, ph = [float(x) for x in patch_annotation.split(' ')]
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

def main():
    label_dict = {'3_pt': [0, 59, 25812], '4_pt': [1, 54, 6275], '5_pt': [2, 47, 750], 'aplite_inclined_pt': [3, 1, 26], 'aplite_vertical_pt': [4, 1, 4], 'bedding_horizontal_pt': [5, 2, 7], 'bedding_inclined_pt': [6, 2, 537], 'bedding_vertical_pt': [7, 3, 42], 'cleavage_inclined_pt': [8, 1, 15], 'cleavage_vertical_pt': [9, 1, 1], 'foliation_horizontal_pt': [10, 1, 42], 'foliation_inclined_pt': [11, 1, 1103], 'foliation_vertical_pt': [12, 1, 43], 'joint_inclined_pt': [13, 1, 97], 'joint_vertical_pt': [14, 1, 19], 'Direction_of_glacier_flow_across_topographic_divide_pt': [15, 1, 21], 'Direction_of_ice_movement_or_meltwater_drainage_over_ice_scoured_bedrock_pt': [16, 1, 10], 'Former_glacial-lake_outlet_or_drainage_diversion_pt': [17, 1, 13], 'Pingo_pt': [18, 1, 47], 'Spring_pt': [19, 1, 2], 'U-shaped_pass_-_where_glacier_crossed_topographic_divide_pt': [20, 1, 4], '2_pt': [21, 53, 4906], 'Ar_Ar_sample_pt': [22, 1, 10], 'paleomag_sample_pt': [23, 1, 56], 'rootless_vent_pt': [24, 1, 42], 'small_cone_pt': [25, 1, 10], 'horiz_bedding_pt': [26, 6, 105], 'inclined_bedding_pt': [27, 13, 3402], 'overturn_bedding_pt': [28, 4, 7], 'USGS_or_Chevron_foss_pt': [29, 1, 940], '1_pt': [30, 32, 403], 'breccia_pipe_pt': [31, 3, 109], 'collapse_structure_pt': [32, 2, 434], 'Cu_or_U_mine_pt': [33, 2, 20], 'dome_pt': [34, 3, 9], 'incl_or_imp_bedding_pt': [35, 2, 1000], 'sinkhole_pt': [36, 6, 1553], 'vertical_joint_pt': [37, 6, 1370], 'volcanic_vent_pt': [38, 2, 59], 'Uranium_mine_pt': [39, 1, 1], 'Uranium_prospect_pt': [40, 1, 2], 'collapse_pt': [41, 4, 152], 'implied_bedding_pt': [42, 1, 134], 'inclined_flow_pt': [43, 1, 11], 'overturned_bedding_pt': [44, 2, 31], 'vert_flow_layering_pt': [45, 1, 3], 'bedding_approx_pt': [46, 1, 35], 'bedding_pt': [47, 3, 1518], 'exp_well_pt': [48, 1, 21], 'horzbed_pt': [49, 2, 29], 'strikedip_pt': [50, 1, 130], 'Estimated_pt': [51, 1, 37], 'Horizontal_pt': [52, 1, 1], 'Inclined_pt': [53, 2, 254], 'horizbed_pt': [54, 1, 46], 'inclbed_pt': [55, 2, 942], 'inclined_pt': [56, 2, 191], 'strike_dip_bedding_pt': [57, 1, 937], 'bedding_overturned_pt': [58, 2, 143], 'well_pt': [59, 2, 13], 'gravel_pit_pt': [60, 1, 8], 'rock_quarry_pt': [61, 1, 1], 'vertical_bedding_pt': [62, 4, 10], 'Strike_and_dip_of_beds_pt': [63, 1, 114], 'Strike_and_dip_of_vertical_beds_pt': [64, 1, 6], 'Strike_and_direction_of_dips_of_beds_from_distant_views_and_photo-interpretations_pt': [65, 1, 155], 'estimated_pt': [66, 1, 19], 'horizontal_pt': [67, 1, 7], 'conodont_pt': [68, 1, 4], 'incl_meta_foliation_pt': [69, 1, 31], 'vert_meta_foliation_pt': [70, 1, 12], 'explor_trench_pt': [71, 1, 4], 'inclin_shear_frac_pt': [72, 1, 2], 'inclined_layering_pt': [73, 1, 3], 'prospect_pit_iron_pt': [74, 1, 24], 'trace_fossils_flathe_pt': [75, 1, 15], 'Inclined_Bedding_pt': [76, 1, 471], 'Vertical_Bedding_pt': [77, 1, 3], 'Overturned_Bedding_pt': [78, 1, 11], 'early_folia_inclin_pt': [79, 1, 83], 'Vertical_Early_Folia_pt': [80, 1, 4], 'late_folia_inclin_pt': [81, 1, 364], 'Vertical_Late_Foliat_pt': [82, 1, 56], 'Phyllonitic_Foliatio_pt': [83, 1, 2], 'Mineral_Lineation_pt': [84, 1, 45], 'inclined_foliation_pt': [85, 1, 119], 'foliation_pt': [86, 1, 4], 'bedding_vertical_top_pt': [87, 1, 3], 'bed_overturn_topknow_pt': [88, 1, 27], 'bedding_top_known_pt': [89, 1, 90], 'sample_site_pt': [90, 1, 22], 'met_vert_foliation_pt': [91, 1, 6], 'metam_foliation_pt': [92, 1, 14], 'mine_pt': [93, 1, 10], 'C14_or_Argon_sample_pt': [94, 2, 12]}

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