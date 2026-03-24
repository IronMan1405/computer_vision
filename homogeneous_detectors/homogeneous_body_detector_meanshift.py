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

        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)

        shifted = cv2.pyrMeanShiftFiltering(img, sp=20, sr=40)

        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            area = cv2.contourArea(cont)

            if area > 800:
                rect = cv2.minAreaRect(cont)
                (x, y), (w, h), angle = rect

                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)

                cv2.polylines(img, [box], True, (0,0,255), 2)
                # cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 2)
                # cv2.putText(img, f"A:{int(area)} R:{int(angle)}deg", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        cv2.imshow("cap", img)
        cv2.imshow("shift", shifted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()