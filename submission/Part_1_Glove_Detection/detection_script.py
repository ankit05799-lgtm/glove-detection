"""Detect gloved vs bare hands in images."""
import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detect gloved vs bare hands in images")
    parser.add_argument("--input", required=True, help="Folder containing .jpg images")
    parser.add_argument("--output", required=True, help="Folder to save annotated images")
    parser.add_argument("--logs", required=True, help="Folder to save JSON detection logs")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()


def get_compliance_status(detections):
    bare_count = sum(1 for d in detections if d["label"] == "bare_hand")
    gloved_count = sum(1 for d in detections if d["label"] == "gloved_hand")
    if bare_count > 0:
        return "VIOLATION", f"{bare_count} bare hand(s) detected - gloves required"
    elif gloved_count > 0:
        return "COMPLIANT", f"{gloved_count} gloved hand(s) detected"
    else:
        return "NO_HANDS", "No hands detected in image"


def draw_banner(image, status, message):
    h, w = image.shape[:2]
    if status == "COMPLIANT":
        color = (0, 255, 0)
    elif status == "VIOLATION":
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)
    cv2.rectangle(image, (0, 0), (w, 40), color, -1)
    cv2.putText(image, f"{status}: {message}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["label"]
        conf = det["confidence"]
        color = (0, 255, 0) if label == "gloved_hand" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    logs_dir = Path(args.logs)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    image_files = sorted(input_dir.glob("*.jpg"))
    if not image_files:
        print(f"No .jpg images found in {input_dir}")
        return

    print(f"Running detection on {len(image_files)} images...")
    for img_path in image_files:
        results = model(str(img_path), conf=args.confidence, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = model.names[cls_id]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                })

        status, message = get_compliance_status(detections)
        log = {
            "filename": img_path.name,
            "compliance_status": status,
            "compliance_message": message,
            "detections": detections
        }
        log_path = logs_dir / (img_path.stem + ".json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        img = cv2.imread(str(img_path))
        if img is not None:
            img = draw_banner(img, status, message)
            if detections:
                img = draw_boxes(img, detections)
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), img)

        print(f"  {img_path.name}: {len(detections)} detections -> {status}")

    print(f"Done. Annotated images: {output_dir}, Logs: {logs_dir}")


if __name__ == "__main__":
    main()
