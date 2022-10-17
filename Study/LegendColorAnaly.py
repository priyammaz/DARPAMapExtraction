import GLP
import json
import cv2 as cv
import os
import matplotlib.pyplot as plt

wmfn = "../../data/training/AK_Bettles.tif" # working map file name
wjfn = "../../data/training/AK_Bettles.json" # working json file name
wmfn = "../../data/training/WI_Wyeville_503636_1958_48000_geo_mosaic.tif"
wjfn = "../../data/training/WI_Wyeville_503636_1958_48000_geo_mosaic.json"

allNames = sorted([i.split(".")[0]  for i in os.listdir("../data/training")
            if ("_poly" not in i) \
            & ("_pt" not in i) \
            & ("_line" not in i) \
            & (".json" not in i)])


name = "WI_Wyeville_503636_1958_48000_geo_mosaic"
name = "WI_WisconsinRapids_503634_1957_48000_geo_mosaic"
name = "NE_PlatteR_2005a"
name = allNames[0] #"AK_Bettles"
for name in allNames:
    wmfn = os.path.join("..", "..", "data", "training", ".".join((name,"tif")))
    wjfn = os.path.join("..", "..", "data", "training", ".".join((name,"json")))
    
    img = cv.imread(wmfn)
    with open(wjfn, "r") as f:
        wjb = json.load(f) #working json blob

    #pt_legend_items = [i for i in wjb["shapes"] if i["label"].split("_")[-1] == "pt"]
    
    #wli = pt_legend_items[2]
    
    #wli = wjb["shapes"][3]
    
    for wli in [i for i in wjb["shapes"] if i["label"].split("_")[-1] == "pt"]:
        
        
        wlp = GLP.getLegendPatch(img, wli)
        
        #pxyz = [tuple(j) for i in wlp for j in i]
        wlp_flat = np.reshape(wlp, (wlp.shape[0]*wlp.shape[1], 3))
        pxyz = wlp_flat
        prgba = [(i[0]/255,i[1]/255, i[2]/255, 1.0) for i in pxyz]
        xyzp = list(zip(*pxyz))
        
        
        fig = plt.figure(figsize=plt.figaspect(1/2.))
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        
        ax.scatter(xyzp[0], xyzp[1], xyzp[2], c=prgba)
    
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(wlp)
        
