import math

def getLegendPatch(img, wli):
    if (wli["points"][0][0] < wli["points"][1][0]) & (wli["points"][0][1] < wli["points"][1][1]):
        x1 = wli["points"][0][0]
        x2 = wli["points"][1][0]
        y1 = wli["points"][0][1]
        y2 = wli["points"][1][1]
    elif (wli["points"][0][0] < wli["points"][1][0]) & (wli["points"][0][1] > wli["points"][1][1]):
        x1 = wli["points"][0][0]
        x2 = wli["points"][1][0]
        y1 = wli["points"][1][1]
        y2 = wli["points"][0][1]
    elif (wli["points"][0][0] > wli["points"][1][0]) & (wli["points"][0][1] < wli["points"][1][1]):
        x1 = wli["points"][1][0]
        x2 = wli["points"][0][0]
        y1 = wli["points"][0][1]
        y2 = wli["points"][1][1]
    else:
        x1 = wli["points"][1][0]
        x2 = wli["points"][0][0]
        y1 = wli["points"][1][1]
        y2 = wli["points"][0][1]

    i1 = math.floor(x1)
    i2 = math.ceil(x2)
    j1 = math.floor(y1)
    j2 = math.ceil(y2)
    
    return img[j1:j2,i1:i2]
