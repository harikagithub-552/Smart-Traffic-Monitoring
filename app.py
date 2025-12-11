from flask import Flask, render_template, request, redirect
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load your YOLO model
model = YOLO("best_simtraffic.pt")  # or yolov8n.pt

# Upload folder
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def count_lanes(img, results):
    h, w, _ = img.shape
    mid_x, mid_y = w//2, h//2

    lanes = [0, 0, 0, 0]

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if cx < mid_x and cy < mid_y: lanes[0] += 1
            elif cx > mid_x and cy < mid_y: lanes[1] += 1
            elif cx < mid_x and cy > mid_y: lanes[2] += 1
            else: lanes[3] += 1

    return lanes

def simulate_signals(lanes):
    out = ""
    for i, c in enumerate(lanes):
        green = round(c / 5, 2)
        out += f"Lane {i+1}: {c} vehicles → Green Time = {green} sec\n"
    return out

@app.route("/", methods=["GET", "POST"])
def index():
    output_img = None
    summary = ""
    signal_output = ""

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read image
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # YOLO detection
        results = model(img_rgb)
        lanes = count_lanes(img_rgb, results)

        # Annotated result
        annotated = results[0].plot()
        output_file = os.path.join(OUTPUT_FOLDER, "result.jpg")
        cv2.imwrite(output_file, annotated)

        # Lane summary
        summary = "\n".join([f"Lane {i+1}: {c} vehicles" for i, c in enumerate(lanes)])

        # Signal simulation
        signal_output = simulate_signals(lanes)

        output_img = "static/output/result.jpg"

    return render_template("index.html",
                           output_img=output_img,
                           summary=summary,
                           signal_output=signal_output)


if __name__ == "__main__":
    app.run(debug=True)
