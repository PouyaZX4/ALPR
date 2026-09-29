import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional

from src.utils.data_types import Detection
from src.utils.onnx_utils import get_onnx_providers


def letterbox(
    img: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True
):
    """
    Resizes and pads image to fit the specified dimensions with preserved aspect ratio.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class OnnxYOLO:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX Model not found at path: {model_path}")

        # Hardware execution providers
        providers = get_onnx_providers()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # GPU / CPU Device Verification
        active_providers = self.session.get_providers()
        primary_device = active_providers[0] if active_providers else "Unknown"
        device_badge = "🚀 [GPU ACTIVE]" if "CUDA" in primary_device or "Dml" in primary_device else "💻 [CPU FALLBACK]"
        print(f"{device_badge} YOLO Model '{os.path.basename(model_path)}' running on: {primary_device}")

    def detect(self, image: np.ndarray, target_classes: Optional[List[int]] = None) -> List[Detection]:
        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]

        # 1. Letterbox resize
        padded_img, ratio, (dw, dh) = letterbox(image, new_shape=(self.imgsz, self.imgsz), auto=False)

        # 2. BGR to RGB, HWC to CHW, Normalize to [0, 1]
        blob = padded_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 3. ONNX Inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        output = outputs[0]  # shape: (1, 84, 8400) or (1, num_boxes, num_attribs)

        # 4. Transpose if output shape is (1, C, N) -> convert to (1, N, C)
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))

        preds = output[0]  # shape: (num_boxes, num_attribs)
        if preds.shape[0] == 0:
            return []

        # 5. Extract bounding boxes and class scores
        # Format: [cx, cy, w, h, class_0, class_1, ...]
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]

        if scores.shape[1] == 1:
            # Single class (e.g. license plate detector)
            class_ids = np.zeros(scores.shape[0], dtype=np.int32)
            confidences = scores[:, 0]
        else:
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

        # Confidence thresholding
        mask = confidences >= self.conf_thresh
        if target_classes is not None:
            mask = mask & np.isin(class_ids, target_classes)

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return []

        # 6. Convert [cx, cy, w, h] to [x1, y1, w, h] for cv2.dnn.NMSBoxes
        x = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        w = boxes_cxcywh[:, 2]
        h = boxes_cxcywh[:, 3]
        boxes_xywh = np.stack([x, y, w, h], axis=1).tolist()

        # 7. Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences.tolist(),
            score_threshold=float(self.conf_thresh),
            nms_threshold=float(self.iou_thresh)
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        # 8. Un-pad and scale bounding boxes back to original image space
        detections: List[Detection] = []
        for idx in indices:
            bx, by, bw, bh = boxes_xywh[idx]
            x1 = (bx - dw) / ratio
            y1 = (by - dh) / ratio
            x2 = (bx + bw - dw) / ratio
            y2 = (by + bh - dh) / ratio

            # Clip within image boundaries
            x1 = max(0.0, min(float(x1), float(orig_w - 1)))
            y1 = max(0.0, min(float(y1), float(orig_h - 1)))
            x2 = max(0.0, min(float(x2), float(orig_w - 1)))
            y2 = max(0.0, min(float(y2), float(orig_h - 1)))

            if (x2 - x1) > 2 and (y2 - y1) > 2:
                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(confidences[idx]),
                    class_id=int(class_ids[idx])
                ))

        return detections