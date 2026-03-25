import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights="DEFAULT"
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

CLASS_COLORS = {
    i: tuple(int(c) for c in np.random.default_rng(i).integers(80, 230, 3))
    for i in range(len(COCO_CLASSES))
}

transform = transforms.Compose([transforms.ToTensor()])


def draw_detections(frame, outputs, threshold=0.5):
    """Draw bounding boxes and labels on a frame."""
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        label_idx = label.item()
        name  = COCO_CLASSES[label_idx]
        color = CLASS_COLORS[label_idx]
        text  = f"{name}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def run_webcam(camera_index=0, threshold=0.5, skip_frames=1):
    """
    Run live object detection on webcam feed.

    Args:
        camera_index: Which camera to use (0 = default webcam)
        threshold:    Confidence threshold for detections
        skip_frames:  Run inference every N frames (1 = every frame)
                      Increase to improve FPS on slow hardware
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit | 'r' to toggle recording | '+'/'-' to adjust threshold")

    frame_count  = 0
    last_outputs = None  # Cache last inference result for skipped frames
    recording    = False
    writer       = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1
        run_inference = (frame_count % max(1, skip_frames) == 0)

        if run_inference:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor  = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                last_outputs = model(tensor)[0]

            last_outputs = {k: v.cpu() for k, v in last_outputs.items()}

        if last_outputs is not None:
            draw_detections(frame, last_outputs, threshold)

        fps_text   = f"FPS skip: 1/{skip_frames}  |  Threshold: {threshold:.2f}"
        rec_text   = "REC" if recording else ""
        cv2.putText(frame, fps_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if recording:
            cv2.putText(frame, rec_text, (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2, cv2.LINE_AA)

        cv2.imshow("Live Object Detection  [q=quit]", frame)

        if recording and writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            threshold = min(0.95, threshold + 0.05)
        elif key == ord('-'):
            threshold = max(0.05, threshold - 0.05)
        elif key == ord('r'):
            recording = not recording
            if recording:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter("detection_output.mp4", fourcc, 20.0, (w, h))
                print("Recording started → detection_output.mp4")
            else:
                if writer:
                    writer.release()
                    writer = None
                print("Recording stopped.")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam(
        camera_index=0,
        threshold=0.5,
        skip_frames=1,
    )