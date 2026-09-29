import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass

from src.utils.plate_format import format_iranian_plate
from src.utils.onnx_utils import get_onnx_providers


@dataclass
class OCRResult:
    plate_text: str
    formatted_text: str
    confidence: float


PERSIAN_LETTERS = set("ابپتثجدسصطعقلمنوهیژآچشظغفکگD")


def estimate_plate_confidence(plate_text: str) -> float:
    """Fallback heuristic confidence scoring based on Iranian plate syntax."""
    if not plate_text:
        return 0.0
    cleaned = plate_text.strip().replace(" ", "").replace("-", "")
    length = len(cleaned)
    if length in (7, 8, 9):
        has_letter = any(c in PERSIAN_LETTERS for c in cleaned)
        digits_count = sum(c.isdigit() or ('۰' <= c <= '۹') for c in cleaned)
        if has_letter and digits_count >= 6:
            return 0.94
        elif digits_count >= 6:
            return 0.88
        return 0.82
    elif 5 <= length <= 10:
        return 0.75
    return 0.50


class OnnxPlateOCR:
    def __init__(
        self,
        onnx_path: str = "models/onnx/crnn_plate_ocr.onnx",
        vocab_path: str = "models/onnx/ocr_vocab.json"
    ):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"OCR model not found at path: {onnx_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"OCR vocabulary not found at path: {vocab_path}")

        providers = get_onnx_providers()
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

        self.blank_label = 0
        self.target_width = 384
        self.target_height = 32
        self.mean = 0.6595
        self.std = 0.1501

        # GPU / CPU Device Verification
        active_providers = self.session.get_providers()
        primary_device = active_providers[0] if active_providers else "Unknown"
        device_badge = "🚀 [GPU ACTIVE]" if "CUDA" in primary_device or "Dml" in primary_device else "💻 [CPU FALLBACK]"
        print(f"{device_badge} OCR Model '{os.path.basename(onnx_path)}' running on: {primary_device}")

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        if plate_crop is None or plate_crop.size == 0:
            return OCRResult(plate_text="", formatted_text="UNKNOWN", confidence=0.0)

        # 1. Convert to Grayscale
        if len(plate_crop.shape) == 3 and plate_crop.shape[2] == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # 2. Resize to CRNN input dimension (width=384, height=32)
        h, w = gray.shape[:2]
        interp = cv2.INTER_AREA if (w > self.target_width or h > self.target_height) else cv2.INTER_CUBIC
        resized = cv2.resize(gray, (self.target_width, self.target_height), interpolation=interp)

        # 3. Horizontal Flip (CRNN model character alignment)
        mirrored = cv2.flip(resized, 1)

        # 4. Normalize to input distribution
        blob = (mirrored.astype(np.float32) / 255.0 - self.mean) / self.std
        blob = blob[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 32, 384)

        # 5. ONNX Inference
        outputs = self.session.run(None, {self.input_name: blob})[0]

        # Standardize logits shape -> (TimeSteps, Classes)
        if outputs.ndim == 3 and outputs.shape[1] == 1:
            logits = outputs[:, 0, :]
        elif outputs.ndim == 3 and outputs.shape[0] == 1:
            logits = outputs[0]
        else:
            logits = outputs

        # 6. Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        preds = np.argmax(probs, axis=-1)
        best_confs = np.max(probs, axis=-1)

        # 7. CTC Greedy Decoding
        char_list = []
        conf_list = []
        prev_idx = -1

        for idx, conf in zip(preds, best_confs):
            if idx != prev_idx:
                if idx != self.blank_label and idx in self.id2label:
                    token = self.id2label[idx]
                    if token and token != ' ':
                        char_list.append(token)
                        conf_list.append(float(conf))
            prev_idx = idx

        raw_mirrored_text = "".join(char_list).strip()

        # 8. Reverse to physical Iranian plate reading order (Left-to-Right)
        true_plate = raw_mirrored_text[::-1]

        # 9. Format Iranian plate structure
        parsed = format_iranian_plate(true_plate)
        formatted_plate = parsed["formatted"]

        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0
        if mean_conf <= 0.0 and len(true_plate) > 0:
            mean_conf = estimate_plate_confidence(true_plate)

        return OCRResult(
            plate_text=true_plate,
            formatted_text=formatted_plate,
            confidence=mean_conf
        )