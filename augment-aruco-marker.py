import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findAruco(img, size=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
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
    dict = os.listdir(path)
    n = len(dict)

    augDict = {}

    for imgpath in dict:
        key = int(os.path.splitext(imgpath)[0])
        print(key)
        augImg = cv2.imread(f'{path}/{key}.jpeg')
        augDict[key] = augImg

    return augDict

def main():
    cap = cv2.VideoCapture(0)

    
    while True:
        success, img = cap.read()

        augDict = loadImages('assets')

        if not success:
            break
        
        aruco = findAruco(img, draw=False)

        if len(aruco[0]) != 0:
            for corners, ids in zip(aruco[0], aruco[1]):
                marker_id = int(ids)
                if marker_id in augDict:
                    # augImg = cv2.imread(f'assets/{marker_id}.jpeg')
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
