import json
import math
import cv2 as cv
import math
import GLP
import matplotlib.pyplot as plt

wmfn = "../../data/training/AK_Bettles.tif" # working map file name
wjfn = "../../data/training/AK_Bettles.json" # working json file name
img = cv.imread(wmfn)
with open(wjfn, "r") as f:
    wjb = json.load(f) #working json blob

pt_legend_items = [i for i in wjb["shapes"] if i["label"].split("_")[-1] == "pt"]

wli = pt_legend_items[3]




wlp = GLP.getLegendPatch(img, wli)


sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(wlp,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=500)

imgKP = cv.drawKeypoints(wlp,kp2,None,(255,0,0),4)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des2,des1,k=1)

    
# Need to draw only good matches, so create a mask
matchesMask = [[1,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
#for i,(m,n) in enumerate(matches):
#    if m.distance < 0.7*n.distance:
#        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
imgKnn = cv.drawMatchesKnn(wlp,kp2,img,kp1,matches,None,**draw_params)

plt.figure()
imgKPax = plt.imshow(imgKP)
plt.figure()
imgKnnax = plt.imshow(imgKnn)

plt.show()

