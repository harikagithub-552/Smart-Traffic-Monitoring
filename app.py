from flask import Flask, render_template, request, redirect
from ultralytics import YOLO
import cv2
import numpy as np
import os
from data_storage import PredictionStorage
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Load YOLO model
model = YOLO("best_simtraffic.pt")

# ------------------ ROBOFLOW CLIENT ------------------

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="XvM2cLH3jpIu3ORAYB8p"   # 🔥 Replace with your new API key
)

# ------------------ FOLDERS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "output")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

storage = PredictionStorage()

# ------------------ LANE COUNT ------------------

def count_lanes(img, results):
    h, w, _ = img.shape
    mid_x, mid_y = w//2, h//2

    lanes = [0, 0, 0, 0]

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if cx < mid_x and cy < mid_y:
                lanes[0] += 1
            elif cx > mid_x and cy < mid_y:
                lanes[1] += 1
            elif cx < mid_x and cy > mid_y:
                lanes[2] += 1
            else:
                lanes[3] += 1

    return lanes

# ------------------ AMBULANCE DETECTION ------------------

def detect_ambulance(filepath, img):
    try:
        # ❌ OLD MODEL (COMMENTED)
        # result = CLIENT.infer(filepath, model_id="ambulance-detection-u4ao4/2")

        # ✅ NEW MODEL
        result = CLIENT.infer(filepath, model_id="on-the-roadv4/1")

    except Exception as e:
        print("Roboflow Error:", e)
        return None, []

    h, w, _ = img.shape
    mid_x, mid_y = w // 2, h // 2

    ambulance_lane = None
    boxes = []

    for pred in result.get("predictions", []):
        if pred["confidence"] > 0.4:

            cx = pred["x"]
            cy = pred["y"]

            # Lane detection
            if cx < mid_x and cy < mid_y:
                ambulance_lane = 0
            elif cx > mid_x and cy < mid_y:
                ambulance_lane = 1
            elif cx < mid_x and cy > mid_y:
                ambulance_lane = 2
            else:
                ambulance_lane = 3

            # Bounding box
            x = int(pred["x"])
            y = int(pred["y"])
            w_box = int(pred["width"])
            h_box = int(pred["height"])

            x1 = int(x - w_box/2)
            y1 = int(y - h_box/2)
            x2 = int(x + w_box/2)
            y2 = int(y + h_box/2)

            boxes.append((x1, y1, x2, y2, pred["class"]))

    return ambulance_lane, boxes


def _is_emergency_vehicle_class(cls_name):
    if not cls_name:
        return False
    s = str(cls_name).lower()
    keywords = ("ambulance", "emergency", "fire truck", "fire engine", "ems", "rescue")
    return any(k in s for k in keywords)


def _boxes_for_ambulance_draw(rf_boxes):
    """Draw only ambulance/emergency boxes; if model is single-class, use all boxes."""
    emergency = [b for b in rf_boxes if _is_emergency_vehicle_class(b[4])]
    if emergency:
        return emergency
    return rf_boxes


def _lane_number_from_box(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    mid_x, mid_y = w // 2, h // 2
    if cx < mid_x and cy < mid_y:
        return 1
    if cx > mid_x and cy < mid_y:
        return 2
    if cx < mid_x and cy > mid_y:
        return 3
    return 4
# ------------------ SIGNAL LOGIC ------------------

def simulate_signals(lanes):
    out = ""
    for i, c in enumerate(lanes):
        if c == 999:
            out += f"🚑 Lane {i+1}: PRIORITY VEHICLE → GREEN\n"
        else:
            green = round(c / 5, 2)
            out += f"Lane {i+1}: {c} vehicles → Green Time = {green} sec\n"
    return out

# ------------------ ROUTE ------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    output_img = None
    summary = ""
    signal_output = ""

    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # YOLO detection
        results = model(img_rgb)
        lanes = count_lanes(img_rgb, results)

        # 🚑 Roboflow detection
        ambulance_lane, rf_boxes = detect_ambulance(filepath, img_rgb)

        if ambulance_lane is not None:
            print(f"🚑 Priority vehicle detected in Lane {ambulance_lane+1}")
            lanes = [0, 0, 0, 0]
            lanes[ambulance_lane] = 999

        # Draw YOLO detections (vehicles)
        annotated = img.copy()

        for r in results:
            for i, box in enumerate(r.boxes.xyxy):
                cls = int(r.boxes.cls[i])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.tolist())

                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(annotated, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Draw Roboflow detections
        for (x1, y1, x2, y2, cls_name) in rf_boxes:
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 4)
            cv2.putText(annotated, cls_name.upper(), (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        # Draw lane division
        h, w, _ = annotated.shape
        cv2.line(annotated, (w//2, 0), (w//2, h), (255,255,0), 2)
        cv2.line(annotated, (0, h//2), (w, h//2), (255,255,0), 2)

        # Save output
        output_file = os.path.join(OUTPUT_FOLDER, "result.jpg")
        cv2.imwrite(output_file, annotated)

        summary = "\n".join([f"Lane {i+1}: {c} vehicles" for i, c in enumerate(lanes)])
        signal_output = simulate_signals(lanes)

        storage.add_prediction(lanes, file.filename, signal_output)

        output_img = "static/output/result.jpg"

    return render_template("predict.html",
                           output_img=output_img,
                           summary=summary,
                           signal_output=signal_output)

# ------------------ OTHER ROUTES ------------------

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    stats = storage.get_statistics()
    recent_predictions = storage.get_recent_predictions()

    lane_distribution = recent_predictions[0]["lanes"] if recent_predictions else [0,0,0,0]
    avg_vehicles = storage.get_average_vehicles_per_image()

    return render_template("dashboard.html",
                           stats=stats,
                           recent_predictions=recent_predictions,
                           lane_distribution=lane_distribution,
                           avg_vehicles=avg_vehicles)


@app.route("/ambulance", methods=["GET", "POST"])
def ambulance_page():
    output_img = None
    ambulance_lane = None
    message = ""

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            message = "No Emergency Vehicle Detected"
            return render_template(
                "ambulance.html",
                output_img=output_img,
                ambulance_lane=ambulance_lane,
                message=message,
            )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        _, rf_boxes = detect_ambulance(filepath, img_rgb)
        boxes_to_draw = _boxes_for_ambulance_draw(rf_boxes)

        annotated = img.copy()
        for (x1, y1, x2, y2, cls_name) in boxes_to_draw:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(
                annotated,
                str(cls_name).upper(),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )

        cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)
        cv2.line(annotated, (0, h // 2), (w, h // 2), (255, 255, 0), 2)

        out_path = os.path.join(OUTPUT_FOLDER, "ambulance_result.jpg")
        cv2.imwrite(out_path, annotated)

        output_img = "static/output/ambulance_result.jpg"

        if boxes_to_draw:
            x1, y1, x2, y2, _ = boxes_to_draw[0]
            lane_num = _lane_number_from_box(x1, y1, x2, y2, w, h)
            ambulance_lane = lane_num
            message = f"🚑 Priority Detected in Lane {lane_num}"
        else:
            message = "No Emergency Vehicle Detected"

    return render_template(
        "ambulance.html",
        output_img=output_img,
        ambulance_lane=ambulance_lane,
        message=message,
    )

# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')