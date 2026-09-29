import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional
from src.utils.data_types import Detection
from src.utils.onnx_utils import get_onnx_providers


class OnnxYOLO:
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45
    ):
        providers = get_onnx_providers()

        sess_options = None
        if hasattr(ort, "SessionOptions"):
            try:
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3
                if 'CPUExecutionProvider' in providers and len(providers) == 1:
                    sess_options.intra_op_num_threads = os.cpu_count() or 4
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            except Exception:
                sess_options = None

        sess_kwargs = {"providers": providers}
        if sess_options is not None:
            sess_kwargs["sess_options"] = sess_options

        self.session = ort.InferenceSession(onnx_path, **sess_kwargs)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, img: np.ndarray, target_classes: Optional[List[int]] = None) -> List[Detection]:
        if img is None or img.size == 0:
            return []

        h_orig, w_orig = img.shape[:2]

        r = min(self.imgsz / h_orig, self.imgsz / w_orig)
        nw, nh = int(round(w_orig * r)), int(round(h_orig * r))
        dw = (self.imgsz - nw) / 2.0
        dh = (self.imgsz - nh) / 2.0

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self.session.run(None, {self.input_name: blob})[0]
        preds = np.transpose(outputs[0])

        boxes_raw = preds[:, :4]
        scores_matrix = preds[:, 4:]

        if target_classes is not None and scores_matrix.shape[1] > 1:
            valid_targets = [c for c in target_classes if c < scores_matrix.shape[1]]
            sub_scores = scores_matrix[:, valid_targets]
            best_sub_idx = np.argmax(sub_scores, axis=1)
            confidences = np.max(sub_scores, axis=1)
            class_ids = np.array(valid_targets)[best_sub_idx]
        elif scores_matrix.shape[1] == 1:
            confidences = scores_matrix[:, 0]
            class_ids = np.zeros(len(confidences), dtype=int)
        else:
            confidences = np.max(scores_matrix, axis=1)
            class_ids = np.argmax(scores_matrix, axis=1)

        mask = confidences >= self.conf_thresh
        if not np.any(mask):
            return []

        boxes_filt = boxes_raw[mask]
        confs_filt = confidences[mask]
        cls_filt = class_ids[mask]

        cx = boxes_filt[:, 0]
        cy = boxes_filt[:, 1]
        w = boxes_filt[:, 2]
        h = boxes_filt[:, 3]

        x1 = np.clip((cx - w / 2.0 - left) / r, 0, w_orig)
        y1 = np.clip((cy - h / 2.0 - top) / r, 0, h_orig)
        x2 = np.clip((cx + w / 2.0 - left) / r, 0, w_orig)
        y2 = np.clip((cy + h / 2.0 - top) / r, 0, h_orig)

        cv_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(x1))]
        cv_confs = [float(c) for c in confs_filt]

        indices = cv2.dnn.NMSBoxes(cv_boxes, cv_confs, self.conf_thresh, self.iou_thresh)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = cv_boxes[i]
                detections.append(Detection(
                    bbox=(float(bx), float(by), float(bx + bw), float(by + bh)),
                    confidence=cv_confs[i],
                    class_id=int(cls_filt[i])
                ))

        return detections