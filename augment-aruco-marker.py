import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findAruco(img, size=6, totalMarkers=250, draw=True):
    '''
    :param img: source image in which to search for aruco markers
    :param size: no. of columns/rows of the aruco marker to detect
    :param totalMarkers: no. of markers in the dictionary
    :param draw: whether to draw contours, pivot, and marker id on detected marker
    :return: list of [corners, ids]
    '''



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key = getattr(aruco, f"DICT_{size}X{size}_{totalMarkers}")
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        if draw:
            aruco.drawDetectedMarkers(img, corners, ids)
    else:
        cv2.putText(img, "no marker detected", (10, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 255, 0), 2)

    return [corners, ids]

def augmentAruco(corners, ids, img, augImg, drawId=True):
    '''
    :param corners: corners of the aruco markers
    :param ids: ids of the aruco markers
    :param img: source image/video capture
    :param augImg: image to be augmented on the marker
    :param drawId: whether to show the id of the marker on the source
    :return: image matrix of the format of source image
    '''



    if ids is not None:
        topleft = corners[0][0][0], corners[0][0][1]
        topright = corners[0][1][0], corners[0][1][1]
        bottomright = corners[0][2][0], corners[0][2][1]
        bottomleft = corners[0][3][0], corners[0][3][1]

        border = np.array([topleft, topright, bottomright, bottomleft])

        h, w, c = img.shape
        augH, augW, augC = augImg.shape

        imgpts = np.array([[0, 0], [augW, 0], [augW, augH], [0, augH]])

        matrix, _ = cv2.findHomography(imgpts, border)

        imgOut = cv2.warpPerspective(augImg, matrix, (w, h))
        cv2.fillConvexPoly(img, border.astype(int), (0,0,0))
        imgOut = img + imgOut

        if drawId:
            cv2.putText(imgOut, f"id: {ids}", (int(topleft[0]), int(topleft[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                
        return imgOut

def loadImages(path):
    '''
    :param path: path in which images are stored with ids as names
    :return: dictionary of images present in the folder with key as id and value as image path
    '''



    dict = os.listdir(path)
    n = len(dict)

    augDict = {}

    for imgpath in dict:
        key = int(os.path.splitext(imgpath)[0])
        augImg = cv2.imread(f'{path}/{imgpath}')
        augDict[key] = augImg

    return augDict

def main():
    cap = cv2.VideoCapture(0)

    augDict = loadImages('assets')
    
    while True:
        success, img = cap.read()

        if not success:
            break
        
        aruco = findAruco(img, draw=False)

        if len(aruco[0]) != 0:
            for corners, ids in zip(aruco[0], aruco[1]):
                marker_id = int(ids.item())
                if marker_id in augDict.keys():
                    augImg = augDict[marker_id]
                    img = augmentAruco(corners, ids, img, augImg)

        cv2.imshow("cam", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
