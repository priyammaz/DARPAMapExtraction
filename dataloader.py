import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from patchify import patchify
import json
from math import floor, ceil
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


Image.MAX_IMAGE_PIXELS = None

class MapPatchLoader(Dataset):
    def __init__(self, 
                 path_to_data, 
                 patch_dim=256, 
                 patch_overlap=0,
                 legend_dim=256):
        
        self.PATH = path_to_data
        self.patch_dim = patch_dim
        self.patch_overlap = patch_overlap
        self.legend_dim = legend_dim
        
        self._grab_unique()
        self._build_patch_dict()
        
    
    def _build_patch_dict(self):
        for key in tqdm(self.metadata):
            path_to_map = os.path.join(self.PATH, key+".tif")
            path_to_json = os.path.join(self.PATH, key+".json")
            
            map_im = np.array(Image.open(path_to_map))
            map_patches = self._patchify_image(map_im, rgb=True)

            with open(path_to_json, "r") as f:
                metadata = json.load(f)["shapes"]
                for meta in metadata:
                    tif_name = key+"_"+meta["label"]+".tif"
                    path_to_tiff = os.path.join(self.PATH, tif_name)
                    tif = np.array(Image.open(path_to_tiff))
                    tif_patches = self._patchify_image(tif, rgb=False)
                    
                    ### GRAB LEGEND LABEL ###
                    legend_label = self._grab_map_legend(map_im, meta["points"])
                    
                    
                    ## GRAB PATCHES WITH SEGMENTATION INFORMATION ###
                    assert(tif_patches.shape == map_patches.shape[:-1])
                    seg_idx = []
                    for idx, patch in enumerate(tif_patches):
                        if patch.sum() > 0:
                            seg_idx.append(idx)
                    
                    tif_chosen_patches = tif_patches[seg_idx]
                    map_chosen_patches = map_patches[seg_idx]
                    
                    plt.imshow(tif_chosen_patches[30])
                    plt.show()
                    
                    plt.imshow(map_chosen_patches[30])
                    plt.show()
                    
                    plt.imshow(legend_label)
                    plt.show()
                    
                    break
                            
        
            break
            
    def _grab_map_legend(self, map_image, points):
        ### CROP OUT LABEL ###
        ix1 = ceil(points[0][0])
        ix2 = floor(points[1][0])
        
        ### CHECK LEFT X IS SMALLER THAN RIGHT X ###
        if ix1 > ix2:
            ix1, ix2 = ix2, ix1

        iy1 = ceil(points[0][1])
        iy2 = floor(points[1][1])
        
        ### CHECK TOP Y SMALLER THAN BOTTOM Y ###
        if iy1 > iy2:
            iy1, iy2 = iy2, iy1
        
        legend_label = map_image[iy1:iy2, ix1:ix2, :]
        
#         y, x, c = legend_label.shape

#         top_pad, bottom_pad = ceil((self.legend_dim - y)/2), floor((self.legend_dim - y)/2)
#         left_pad, right_pad = ceil((self.legend_dim - x)/2), floor((self.legend_dim - x)/2)
        
#         legend_label = np.pad(legend_label, 
#                               (
#                                   (top_pad, bottom_pad),
#                                   (left_pad, right_pad),
#                                   (0, 0)
#                               ),
#                                mode='constant', constant_values=255) 

        legend_label = cv2.resize(legend_label, 
                                  dsize=(self.legend_dim , self.legend_dim ),
                                  interpolation=cv2.INTER_LINEAR)

        return legend_label
    
    def _patchify_image(self, map_image, rgb=False):
        
        patch_tuple = [self.patch_dim, self.patch_dim]
        
        if rgb:
            patch_tuple.append(3)

        step = self.patch_dim - self.patch_overlap

        return patchify(map_image, tuple(patch_tuple), step=step).reshape(-1, *patch_tuple)
        
        
            
            
    def _grab_unique(self):
        all_files = sorted(os.listdir(self.PATH))
        self.metadata = [file[:-5] for file in all_files if ".json" in file]
        
class MAPLoader(Dataset):
    def __init__(self, PATH_TO_DATA):
        self.PATH = PATH_TO_DATA
        
        ### LOAD IN PATH DATA TO ALL SAMPLES ###
        self._grab_paths()
        
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, index):
        key, segmentation = self.metadata_df.iloc[index, :]
        full_map, legend_label = self._grab_map_legend(key, segmentation)
        legend_name = segmentation[:-4]
        segmentation_map = np.array(Image.open(os.path.join(self.PATH, key + "_" + segmentation)))
        
        return full_map, segmentation_map, legend_label, legend_name
        
    
    def _grab_map_legend(self, key, segmentation):
        ### GRAB POINTS SURROUNDING LEGEND LABEL ###
        with open(os.path.join(self.PATH, key+'.json'), 'r') as f:
            meta_json = json.load(f)
        
        for i in meta_json["shapes"]:
            if i["label"] == segmentation[:-4]:
                points = i["points"]
        full_map = np.array(Image.open(os.path.join(self.PATH, key+".tif")))
            
        ### CROP OUT LABEL ###
        ix1 = ceil(points[0][0])
        ix2 = floor(points[1][0])

        iy1 = ceil(points[0][1])
        iy2 = floor(points[1][1])

        legend_label = full_map[iy1:iy2, ix1:ix2, :]
        
        return full_map, legend_label            
        
    def _grab_paths(self):
        all_files = sorted(os.listdir(self.PATH))
        unique_jsons = [file[:-5] for file in all_files if ".json" in file]
        metadatas = {"key": [],
                     "segmentation": []}
        for unique in unique_jsons:
            segmentation = [file.replace(unique+"_","") for file in all_files if unique in file][2:]
            key = [unique for i in range(len(segmentation))]
            metadatas["segmentation"].extend(segmentation)
            metadatas["key"].extend(key)
            
        self.metadata_df = pd.DataFrame(metadatas)

if __name__ == "__main__":
    PATH_TO_DATA = "/home/shared/DARPA/training"
    dataset = MAPLoader(PATH_TO_DATA)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=2)
    
    for data in dataloader:
        print(data)