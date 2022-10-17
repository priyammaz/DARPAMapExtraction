import GLP
import json
import cv2 as cv
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

wmfn = "../../data/training/AK_Bettles.tif" # working map file name
wjfn = "../../data/training/AK_Bettles.json" # working json file name
wmfn = "../../data/training/WI_Wyeville_503636_1958_48000_geo_mosaic.tif"
wjfn = "../../data/training/WI_Wyeville_503636_1958_48000_geo_mosaic.json"

allNames = sorted([i.split(".")[0]  for i in os.listdir("../../data/training")
            if ("_poly" not in i) \
            & ("_pt" not in i) \
            & ("_line" not in i) \
            & (".json" not in i)])


name = "WI_Wyeville_503636_1958_48000_geo_mosaic"
name = "WI_WisconsinRapids_503634_1957_48000_geo_mosaic"
name = "NE_PlatteR_2005a"
name = allNames[0] #"AK_Bettles"

wmfn = os.path.join("..", "..", "data", "training", ".".join((name,"tif")))

wjfn = os.path.join("..", "..", "data", "training", ".".join((name,"json")))

img = cv.imread(wmfn)
with open(wjfn, "r") as f:
    wjb = json.load(f) #working json blob

wli = wjb["shapes"][2]
wlfn = os.path.join("..", "..", "data", "training", ".".join(("_".join((name, wli["label"])),"tif")))
img_label = cv.imread(wlfn)
wlp = GLP.getLegendPatch(img, wli)

def colorAppleCore(wlp_flat, bRange=[75,75,75], wRange=[235, 235, 235], greyR=10):

    _bRmask = wlp_flat > bRange
    bRmask = np.apply_along_axis(all, 1, _bRmask)
    _wRmask = wlp_flat < wRange
    wRmask = np.apply_along_axis(all, 1, _wRmask)
    
    wlp_f_units = np.ones_like(wlp_flat)

    dist_F_cross = np.linalg.norm(np.cross(wlp_flat, wlp_f_units), axis=1)

    #dist_F_norm = np.linalg.norm(wlp_f_units - wlp_flat, axis=1)
    dist_F_norm = np.linalg.norm(wlp_f_units, axis=1)

    dist_F = dist_F_cross/dist_F_norm

    gDistMask = dist_F > greyR

    return wlp_flat[gDistMask & bRmask & wRmask,:]

def getColorBBPlots(wlp_flat, bRange=[75,75,75], wRange=[235, 235, 235], greyR=10):

    wlp_cored = colorAppleCore(wlp_flat, bRange=bRange, wRange=wRange, greyR=greyR)

    if len(wlp_cored) > 0:
        pxyz = wlp_cored
    else:
        pxyz = wlp_flat
        
    prgba = [(i[0]/255,i[1]/255, i[2]/255, 1.0) for i in pxyz]
    xyzp = list(zip(*pxyz))

    fig = plt.figure(figsize=plt.figaspect(1/2.))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    ax.scatter(xyzp[0], xyzp[1], xyzp[2], c=prgba)
    ax.scatter([0,255], [0,255], [0,255],c="r", marker="x")
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(wlp)




wlp_flat = np.array(np.reshape(wlp, (wlp.shape[0]*wlp.shape[1], 3)), dtype=float)
wlp_cored = colorAppleCore(wlp_flat, bRange=[75,75,75], wRange=[235, 235, 235], greyR=10)
try_n = 4
kmeans = []
for i in range(0,4):
    kmeans.append( KMeans(n_clusters=1, random_state=0).fit_predict(wlp_cored))

total_classes = len(set([k2 for k in kmeans for k2 in set(k)]))

if total_classes == 1:
    CovM = np.cov(list(zip(*wlp_cored)))
    MeanV = np.mean(wlp_cored, axis=0)
    mv = multivariate_normal(MeanV, CovM)

    cutoff = min(mv.pdf(wlp_cored))

    like_raster = mv.pdf(img)
elif total_classes > 1:
    pass
else:
    pass


def makeBlackFromGrey(img, wlp_flat, GreyCutoff = [150, 150, 150], maxLikeFac=1/4):
    """
    This function takes in the image, `img` and a single legend patch flattend to `[:, 3]` and reruns
    an image with the same HxW as img but a single chanel that is 0 where all the black-ish pixles were
    and 255 everywhere else
    """
    greyClass = wlp_flat[np.apply_along_axis(all, 1, wlp_flat < GreyCutoff)]
    
    CovM = np.cov(list(zip(*greyClass)))
    MeanV = np.mean(greyClass, axis=0)
    mvGrey = multivariate_normal(MeanV, CovM)
    
    greyLike_raster = mvGrey.pdf(img)
    max_like = np.max(greyLike_raster)
    
    blackMap = np.full_like(greyLike_raster, 255, dtype=np.uint8)

    max_mask = greyLike_raster > max_like*maxLikeFac

    blackMap[max_mask] = 0

    return blackMap


blackWLP = makeBlackFromGrey(wlp, wlp_flat, GreyCutoff = [175, 175, 175], maxLikeFac=1/8)

blackMap = makeBlackFromGrey(img, wlp_flat, GreyCutoff = [175, 175, 175], maxLikeFac=1/8)


# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(blackMap,None)
kp2, des2 = orb.detectAndCompute(blackWLP,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des2,des1)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
Orbimg = cv.drawMatches(blackWLP,kp2,blackMap,kp1,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(Orbimg),plt.show()



img_s = img[:, 1:6000,:]
blackMap = makeBlackFromGrey(img_s, wlp_flat, GreyCutoff = [175, 175, 175], maxLikeFac=1/8)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(blackMap,None)
kp2, des2 = sift.detectAndCompute(blackWLP,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 25)
search_params = dict(checks=50)

imgKP = cv.drawKeypoints(blackWLP,kp2,None,(255,0,0),4)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des2,des1,k=2)


# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
imgKnn = cv.drawMatchesKnn(blackWLP,kp2,blackMap,kp1,matches,None,**draw_params)

plt.figure()
imgKPax = plt.imshow(imgKP)
plt.figure()
imgKnnax = plt.imshow(imgKnn)

plt.show()
