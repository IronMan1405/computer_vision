import cv2
import cv2.aruco as aruco

def findAruco(img, size=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # key = getattr(aruco, f'DICT_{size}X{size}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejected = detector.detectMarkers(gray)
    # print(f"corners: {corners}, ids: {ids}, rejected: {len(rejected)}")

    if ids is not None:
        # print(corners)
        # cv2.putText(img, ids, (corners[0]), cv2.FONT_HERSHEY_DUPLEX, 3, (0,255,0), 2)
        if draw:
            aruco.drawDetectedMarkers(img, corners, ids)
    else:
        cv2.putText(img, "no marker detected", (10, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        s2, img2 = cap2.read()

        if not success:
            break
        if not s2:
            break
        
        findAruco(img)
        # cv2.imshow("cam", img)

        findAruco(img2)
        cv2.imshow("cam2", img2)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
