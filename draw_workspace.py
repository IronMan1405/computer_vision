import cv2
import numpy as np
from find_aruco_marker import findAruco
from drawcontours_aruco import drawArucoContours
from calculate_cm_px_ratio import calculateLength, calculateRatio, actualSideLength

def get_workspace_corners(img, cnr0, cnr1, cnr2):
    [corners, ids] = findAruco(img, draw=False)
    
    if ids is not None:
        for cnr, id in zip(corners, ids):
            marker_id = int(id.item())
            if marker_id == 5:
                cnr0 = tuple(map(int, cnr[0][0]))
            if marker_id == 42:
                cnr1 = tuple(map(int, cnr[0][1]))
            if marker_id == 249:
                cnr2 = tuple(map(int, cnr[0][0]))

    return cnr0, cnr1, cnr2


def draw_workspace(img, cnr0, cnr1, cnr2):
    if cnr0 is not None:
        cv2.circle(img, cnr0, 2, (0,0,127), 2)
    if cnr1 is not None:
        cv2.circle(img, cnr1, 2, (0,0,127), 2)
    if cnr2 is not None:
        cv2.circle(img, cnr2, 2, (0,0,127), 2)

    if cnr0 is not None and cnr1 is not None and cnr2 is not None:
        cv2.rectangle(img, cnr1, cnr2, 255, 2)

        pts = np.array([cnr0, cnr1, cnr2], dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        r = cv2.arcLength(np.array([cnr1, cnr2]), False)
        cv2.circle(img, cnr2, int(r), (0,127,0), 2)


def main():
    cap = cv2.VideoCapture(0)

    cn0 = None
    cn1 = None
    cn2 = None

    while True:
        success, img = cap.read()

        if not success:
            break

        [corners, ids] = findAruco(img, draw=False)
        
        cn0, cn1, cn2 = get_workspace_corners(img, cn0, cn1, cn2)
        draw_workspace(img, cn0, cn1, cn2)

        ratio = 0
        appL = 0
        
        if len(corners) != 0:
            for corner, id in zip(corners, ids):
                marker_id = int(id.item())
                if marker_id == 5:
                    drawArucoContours([corner], [marker_id], img)
                    appL = calculateLength(marker_id, corner)
                    ratio = calculateRatio(actualSideLength, appL)

        #     cv2.putText(img, f"{appL * ratio:.2f}", (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        # cv2.putText(img, f"{appL: .2f} px", (10, 15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        # cv2.putText(img, f"cm to pixel ratio: {ratio: .2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        cv2.imshow("cam", img)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()