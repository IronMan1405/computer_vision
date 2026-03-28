import cv2
import cv2.aruco as aruco

def findAruco(img, size=6, totalMarkers=250, draw=True):
    '''
    :param img: source image in which to search for aruco markers
    :param size: no. of columns/rows of the aruco marker to detect
    :param totalMarkers: no. of markers in the dictionary
    :param draw: whether to draw contours, pivot, and marker id on detected marker
    :return: list of [corners, ids]
    '''
    
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key = getattr(aruco, f'DICT_{size}X{size}_{totalMarkers}')
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

def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        if not success:
            break
        
        [corners, ids] = findAruco(img)
        cv2.imshow("cam", img)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
