import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from patchify import patchify
import pandas as pd
from math import floor, ceil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2
from collections import Counter

Image.MAX_IMAGE_PIXELS = None


class GenMapPatches:
    def __init__(self, 
                 path_to_data,
                 path_to_store,
                 patch_dim=256, 
                 patch_overlap=0,
                 legend_dim=256,
                 skip_train=None, 
                 skip_val=None):
        
        self.PATH = path_to_data
        self.PATH_TO_STORE = path_to_store
        self.patch_dim = patch_dim
        self.patch_overlap = patch_overlap
        self.legend_dim = legend_dim
        self.skip_train = skip_train
        self.skip_val = skip_val
        
        self._make_dir()
        self._gen_patches()
        
    def _make_dir(self):
        folder_name = f"patch_size_{self.patch_dim}_patch_overlap_{self.patch_overlap}_legend_size_{self.legend_dim}"
        path_to_folder = os.path.join(self.PATH_TO_STORE, folder_name)
        path_to_train = os.path.join(path_to_folder, "training")
        path_to_val = os.path.join(path_to_folder, "validation")
        
        self.folder_directories = {"training":  {"labels": os.path.join(path_to_train, "labels"), 
                                            "map_patches": os.path.join(path_to_train, "map_patches"),
                                            "seg_patches": os.path.join(path_to_train, "seg_patches")},
                              
                                  "validation": {"labels": os.path.join(path_to_val, "labels"), 
                                                 "map_patches": os.path.join(path_to_val, "map_patches"),
                                                 "seg_patches": os.path.join(path_to_val, "seg_patches")}
                                  }
        
    
        
        for path in [path_to_folder, path_to_train, path_to_val]:
            try: 
                os.mkdir(path) 
            except OSError as error: 
                pass
                
        for directory in ["training", "validation"]:
            for folder in ["labels", "map_patches", "seg_patches"]:
                try:
                    os.mkdir(self.folder_directories[directory][folder])
                except OSError as error:
                    pass
            
        
    def _gen_patches(self):
        for directory in ["training", "validation"]:
            print(f"Parsing {directory} data")
            self._grab_unique(os.path.join(self.PATH, directory))

            for idx, key in enumerate(self.metadata):
                print(f"Patching {idx}/{len(self.metadata)}: {key}")
                path_to_map = os.path.join(self.PATH, directory, key+".tif")
                path_to_json = os.path.join(self.PATH, directory, key+".json")

                map_im = np.array(Image.open(path_to_map))
                map_patches = self._patchify_image(map_im, rgb=True)

                with open(path_to_json, "r") as f:
                    metadata = json.load(f)["shapes"]
                    for idx, meta in enumerate(metadata):
                        
                        print(f"{idx}/{len(metadata)}", end='\r')
                        tif_name = key+"_"+meta["label"]+".tif"
                        path_to_tiff = os.path.join(self.PATH, directory, tif_name)
                        tif = np.array(Image.open(path_to_tiff))
                        tif_patches = self._patchify_image(tif, rgb=False)

                        ### GRAB LEGEND LABEL ###
                        try:
                        ### IF LEGEND LABEL ISNT EMPTY ###
                            legend_label = self._grab_map_legend(map_im, meta["points"])

                            ## GRAB PATCHES WITH SEGMENTATION INFORMATION ###
                            assert(tif_patches.shape == map_patches.shape[:-1])
                            seg_idx = []
                            for idx, patch in enumerate(tif_patches):
                                if patch.sum() > 0: # If any segmentation information exists in patch
                                    idx_map = Image.fromarray(map_patches[idx])
                                    idx_seg = Image.fromarray(tif_patches[idx] * 255)
                                    idx_label = Image.fromarray(legend_label)

                                    label_name = meta["label"]
                                    filename = f"{key}_{label_name}_{idx}"

                                    map_save_path = os.path.join(self.folder_directories[directory]["map_patches"], f"{filename}.png")
                                    seg_save_path = os.path.join(self.folder_directories[directory]["seg_patches"], f"{filename}.png")
                                    label_save_path = os.path.join(self.folder_directories[directory]["labels"], f"{filename}.png")

                                    idx_map.save(map_save_path, format="png")
                                    idx_seg.save(seg_save_path, format="png")
                                    idx_label.save(label_save_path, format="png")

                        except Exception as e:
                            print(f"[LOGGING]: Key:{key}, Meta:{label_name} Label is Broken")
                                
    
            
    def _grab_map_legend(self, map_image, points):
        points = np.array(points)
        ix1, iy1 = list(points.min(axis=0).astype(int))
        ix2, iy2 = list(points.max(axis=0).astype(int))
        
        legend_label = map_image[iy1:iy2, ix1:ix2, :]

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
        
        
            
    def _grab_unique(self, directory):
        all_files = sorted(os.listdir(directory))
        self.metadata = [file[:-5] for file in all_files if ".json" in file]

        if "training" in directory and self.skip_train is not None:
            print(f"Skipping Until Map {self.skip_train}")
            start_idx = self.metadata.index(self.skip_train)
            self.metadata = self.metadata[start_idx:]
           
        if "validation" in directory and self.skip_val is not None:
            print(f"Skipping Until Map {self.skip_val}")
            start_idx = self.metadata.index(self.skip_train)
            self.metadata = self.metadata[start_idx:]
            
if __name__ == "__main__":
    PATH_TO_DATA = "/home/shared/DARPA"
    PATH_TO_STORE = os.path.join(PATH_TO_DATA, "patched_data")
    dataset = GenMapPatches(path_to_data=PATH_TO_DATA, path_to_store=PATH_TO_STORE)
