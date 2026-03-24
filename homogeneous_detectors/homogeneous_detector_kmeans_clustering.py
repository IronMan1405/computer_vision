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

cap = cv2.VideoCapture(0)
kernel = np.ones((9, 9), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        break

    img   = cv2.resize(img, (640, 480))
    blur  = cv2.GaussianBlur(img, (7, 7), 0)
    masks = get_dominant_color_masks(blur, k=4)

    for mask, color in masks:
        # Clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            area = cv2.contourArea(cont)
            if area < 5000:
                continue

            rect          = cv2.minAreaRect(cont)
            (x, y), _, angle = rect
            box           = cv2.boxPoints(rect).astype(np.intp)

            # Use the cluster's actual color for the box
            bgr = tuple(int(c) for c in color)
            cv2.polylines(img, [box], True, bgr, 2)
            cv2.circle(img, (int(x), int(y)), 4, bgr, -1)
            cv2.putText(img, f"A:{int(area)} R:{int(angle)}d",
                        (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)

    cv2.imshow("cap", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()