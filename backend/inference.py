"""
SONARIS - inference wrapper (ONNX Runtime edition).

Runs the two-model pipeline (main multiclass detector + pipeline specialist)
using onnxruntime instead of PyTorch/Ultralytics. This exists purely to cut
the serving process's memory footprint - PyTorch's baseline C++ runtime alone
is ~150-300MB before a single weight is loaded, which was pushing a 512MB
deploy over the limit. onnxruntime has none of that overhead.

The .pt -> .onnx export is a ONE-TIME step, done locally/offline with:
    yolo export model=best.pt format=onnx imgsz=640 simplify=True
It needs ultralytics+torch, but only at export time - never at serve time.
This module does NOT import torch or ultralytics at all.

Preprocessing (letterbox resize), decoding, and NMS are implemented by hand
below since we no longer have Ultralytics' Results wrapper doing it for us.
Both exported models are YOLOv8-family output layout: [1, 4+nc(+32 mask
coeffs for segmentation), 8400]. Mask coefficients (if present) are ignored -
this app only ever needed bounding boxes, never pixel masks.
"""
import argparse
import os
import yaml
import numpy as np
import onnxruntime as ort
from PIL import Image

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def letterbox(image: Image.Image, size: int = 640):
    """Resize with aspect ratio preserved, padding to a square. Returns the
    padded array plus the scale/offset needed to map boxes back to original
    image coordinates."""
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = image.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW
    return arr, scale, pad_x, pad_y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """Greedy NMS. boxes: (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class OnnxYoloModel:
    """One ONNX YOLO head (detect or segment - mask output, if any, is ignored)."""

    def __init__(self, path: str):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image, conf: float, iou: float, imgsz: int):
        arr, scale, pad_x, pad_y = letterbox(image, imgsz)
        outputs = self.session.run(None, {self.input_name: arr.astype(np.float32)})
        pred = outputs[0][0]  # (attrs, 8400) - box output is always outputs[0]

        pred = pred.T  # -> (8400, attrs): [x,y,w,h, class_scores..., mask_coeffs...]
        # Number of classes is inferred by the caller via class_names length;
        # here we just take every score column after the first 4 as a class
        # score up to num_classes, then stop (ignore trailing mask coeffs).
        return pred, scale, pad_x, pad_y

    def decode(self, pred, num_classes: int, scale, pad_x, pad_y, conf: float, iou: float):
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:4 + num_classes]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= conf
        if not np.any(mask):
            return []
        boxes_xywh, class_ids, confidences = boxes_xywh[mask], class_ids[mask], confidences[mask]

        # center xywh (in letterboxed 640-space) -> xyxy, then undo letterbox
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = nms(xyxy, confidences, iou)
        return [
            {"class_id": int(class_ids[i]), "confidence": float(confidences[i]), "bbox": xyxy[i]}
            for i in keep
        ]


class InferenceEngine:
    def __init__(self, config=None):
        self.config = config or load_config()
        base_dir = os.path.dirname(__file__)

        self.multiclass_names = self.config["classes"]["multiclass"]
        self.pipeline_names = self.config["classes"]["pipeline_specialist"]

        self.multiclass_model = self._safe_load(
            os.path.join(base_dir, self.config["models"]["multiclass"]), "multiclass"
        )
        self.pipeline_model = self._safe_load(
            os.path.join(base_dir, self.config["models"]["pipeline_specialist"]), "pipeline_specialist"
        )

    def _safe_load(self, path, name):
        if not os.path.isfile(path):
            print(f"[WARNING] {name} model not found at {path} - "
                  f"that head will return no detections until the .onnx file is placed there.")
            return None
        model = OnnxYoloModel(path)
        print(f"[OK] Loaded {name} ONNX model from {path}")
        return model

    def _run_one(self, model, class_names, image, conf, iou, imgsz, source_tag):
        if model is None:
            return []
        pred, scale, pad_x, pad_y = model.predict(image, conf, iou, imgsz)
        raw = model.decode(pred, len(class_names), scale, pad_x, pad_y, conf, iou)
        out = []
        for r in raw:
            class_id = r["class_id"]
            class_name = class_names.get(class_id, f"unknown-{class_id}")
            x1, y1, x2, y2 = [float(v) for v in r["bbox"]]
            out.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(r["confidence"], 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "source": source_tag,
            })
        return out

    def run(self, image, conf=None, pipeline_conf=None, iou=None, imgsz=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        cfg = self.config["inference"]
        conf = cfg["confidence"] if conf is None else conf
        pipeline_conf = cfg["pipeline_confidence"] if pipeline_conf is None else pipeline_conf
        iou = cfg["iou"] if iou is None else iou
        imgsz = cfg["imgsz"] if imgsz is None else imgsz

        detections = self._run_one(
            self.multiclass_model, self.multiclass_names, image, conf, iou, imgsz, "multiclass"
        )
        pipeline_specialist = self._run_one(
            self.pipeline_model, self.pipeline_names, image, pipeline_conf, iou, imgsz, "pipeline_specialist"
        )
        return {"detections": detections, "pipeline_specialist": pipeline_specialist}


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def run_inference(image, conf=None, pipeline_conf=None, iou=None, imgsz=None):
    return get_engine().run(image, conf=conf, pipeline_conf=pipeline_conf, iou=iou, imgsz=imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONARIS - standalone sonar inference (ONNX Runtime)")
    parser.add_argument("image_path")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--pipeline-conf", type=float, default=None)
    args = parser.parse_args()

    result = run_inference(args.image_path, conf=args.conf, pipeline_conf=args.pipeline_conf)
    result["image"] = os.path.basename(args.image_path)

    import json
    print(json.dumps(result, indent=2))
