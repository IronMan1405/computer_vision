import cv2
import numpy as np

prototext = 'mobilenetssd/deploy.prototxt'
model = 'mobilenetssd/mobilenet_iter_73000.caffemodel'
confidenceThresh = 0.5

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

colors = np.random.uniform(0, 255, size=(len(classes), 3)) #generates random colors of the form (r,g,b)

net = cv2.dnn.readNetFromCaffe(prototext, model)
print("Loaded OK!")

cap = cv2.VideoCapture(0)

def main():
    
    while True:
        success, img = cap.read()

        if not success:
            return
        
        h, w, c = img.shape
        resized = cv2.resize(img, (300, 300))
        blob = cv2.dnn.blobFromImage(resized, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True)
        
        net.setInput(blob)

        detected = net.forward()

        for i in range(detected.shape[2]):
            confidence = detected[0][0][i][2]
            
            if confidence >= confidenceThresh:
                class_index = int(detected[0][0][i][1])
                
                tlx, tly = int(detected[0][0][i][3] * w), int(detected[0][0][i][4] * h)
                brx, bry = int(detected[0][0][i][5] * w), int(detected[0][0][i][6] * h)

                label = f"{classes[class_index]}: {confidence:.2f}"
                color = colors[class_index].tolist()

                cv2.rectangle(img, (tlx, tly), (brx, bry), color, 3)
                cv2.putText(img, label, (tlx, max(tly - 10, 10)), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)
        
        cv2.imshow("capture", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()