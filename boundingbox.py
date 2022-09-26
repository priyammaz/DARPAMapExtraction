import cv2
import numpy as np

def bounding_box(imarray):
    """
    imarray: from cv2.imread(filePath)
    """
    ## EDGE DETECTION
    gray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    ## CONTOUR PLOT
    con = np.zeros_like(imarray) # Blank canvas.
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    ## FIND THE CORNER POINTS
    con = np.zeros_like(imarray)   # Blank canvas.
    # Loop over the contours.
    for c in page:
        epsilon = 0.02 * cv2.arcLength(c, True)  # Approximate the contour.
        corners = cv2.approxPolyDP(c, epsilon, True)

        if len(corners) == 4:  # If our approximated contour has four points
              break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    print(corners)
    xmin, xmax = min(point[0] for point in corners), max(point[0] for point in corners)
    ymin, ymax = min(point[1] for point in corners), max(point[1] for point in corners)
    return [xmin, ymin, xmax, ymax]
