import cv2
import numpy as np

def get_dominant_color_masks(img, k=4):
    """Segment image into k color clusters, return mask for each."""
    data = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    labels  = labels.flatten()

    masks = []
    for i in range(k):
        mask = (labels == i).reshape(img.shape[:2]).astype(np.uint8) * 255
        masks.append((mask, centers[i]))  # mask + its dominant color

    return masks

class HomogeneousDetector:
    def hsv_detect(self, src, lower=None, upper=None, kernel_size=(5,5), draw_boxes=True, draw_text=True, draw_center=True):
        # H, S, V lower and upper bounds
        if lower is None:
            lower = np.array([0, 50, 50])
        if upper is None:
            upper = np.array([40, 255, 255])

        img = cv2.resize(src, (640, 480), interpolation=cv2.INTER_LINEAR)
        img_blur = cv2.GaussianBlur(img, (7, 7), 0)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)


        kernel = np.ones(kernel_size, np.uint8)

        mask = cv2.inRange(img_hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # remove/ignore small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # fill holes

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            area = cv2.contourArea(cont)

            if area > 1000:
                rect = cv2.minAreaRect(cont)
                (x, y), (w, h), angle = rect

                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)

                if draw_boxes:
                    cv2.polylines(img, [box], True, (0,0,255), 2)
                if draw_center:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 2)
                if draw_text:
                    cv2.putText(img, f"A:{int(area)} R:{int(angle)}deg", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return img
    
    def mean_shift_detect(self, src, sp=20, sr=40, draw_boxes=True, draw_text=True, draw_center=True):
        img = cv2.resize(src, (320, 240), interpolation=cv2.INTER_LINEAR)

        shifted = cv2.pyrMeanShiftFiltering(img, sp=sp, sr=sr)

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

                if draw_boxes:
                    cv2.polylines(img, [box], True, (0,0,255), 2)
                if draw_center:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 2)
                if draw_text:
                    cv2.putText(img, f"A:{int(area)} R:{int(angle)}deg", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        return img
    
    def kmeans_detect(self, src, k=4, kernel_size=(9, 9), boxColor=None, draw_boxes=True, draw_text=True, draw_center=True):
        img = cv2.resize(src, (640, 480))
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        kernel = np.ones(kernel_size, np.uint8)
        masks = get_dominant_color_masks(blur, k=k)

        for mask, color in masks:
            # Clean mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cont in contours:
                area = cv2.contourArea(cont)
                if area < 5000:
                    continue

                rect = cv2.minAreaRect(cont)
                (x, y), _, angle = rect
                box = cv2.boxPoints(rect).astype(np.intp)

                # Use the cluster's actual color for the box
                if boxColor is None:
                    box_color = tuple(int(c) for c in color)
                else:
                    box_color = boxColor

                if draw_boxes:
                    cv2.polylines(img, [box], True, box_color, 2)
                if draw_center:
                    cv2.circle(img, (int(x), int(y)), 4, box_color, -1)
                if draw_text:
                    cv2.putText(img, f"A:{int(area)} R:{int(angle)}d", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        return img