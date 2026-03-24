import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found")
        return

    while True:
        success, img = cap.read()
                
        if not success:
            break

        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        img_blur = cv2.GaussianBlur(img, (7, 7), 0)

        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        # H, S, V lower and upper bounds
        lower = np.array([0, 50, 50])
        upper = np.array([40, 255, 255])

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.inRange(img_hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # remove/ignore small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # fill holes

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            area = cv2.contourArea(cont)

            if area > 1000:
                # x, y, w, h = cv2.boundingRect(cont)
                rect = cv2.minAreaRect(cont)
                (x, y), (w, h), angle = rect

                # cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

                # cv2.drawContours(img, [cont], -1, (0, 255, 0), 3)

                # cv2.polylines(img, [cont], True, (0, 0, 255), 2)
                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)

                cv2.polylines(img, [box], True, (0,0,255), 2)

                cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 2)

                cv2.putText(img, f"A:{int(area)} R:{int(angle)}deg", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        cv2.imshow("cap", img)
        cv2.imshow("mask", mask)
        # cv2.imshow("hsv", img_hsv)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()