import os
import json
import numpy as np
import pandas as pd
from math import floor, ceil
from PIL import Image
from torch.utils.data import Dataset, DataLoader

Image.MAX_IMAGE_PIXELS = None

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
    dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=16)