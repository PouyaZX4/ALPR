import re
import cv2
import numpy as np
import huggingface_hub

# Patch missing Repository in newer huggingface_hub
if not hasattr(huggingface_hub, "Repository"):
    class DummyRepository:
        pass
    huggingface_hub.Repository = DummyRepository

from hezar.models import Model
from dataclasses import dataclass


@dataclass
class OCRResult:
    plate_text: str
    confidence: float


# Common Persian letters on standard plates
PERSIAN_LETTERS = set("ابپتثجدسصطعقلمنوهیژ")


def estimate_plate_confidence(plate_text: str) -> float:
    """
    Fallback confidence heuristic based on Iranian plate structural validity
    Standard format: 2 digits + 1 letter + 3 digits + 2 digits (e.g. 12ب34567) -> 8-9 chars
    """
    if not plate_text:
        return 0.0

    cleaned = plate_text.strip().replace(" ", "")
    length = len(cleaned)

    # Standard Iranian private car plate length is 8 characters (or 9 if formatted)
    if length in (8, 9):
        # Check if contains letters and digits
        has_letter = any(c in PERSIAN_LETTERS for c in cleaned)
        digits_count = sum(c.isdigit() or ('۰' <= c <= '۹') for c in cleaned)

        if has_letter and digits_count >= 6:
            return 0.94
        elif digits_count >= 7:
            return 0.88
        return 0.82
    elif 6 <= length <= 10:
        return 0.75
    else:
        return 0.50


class PlateOCR:
    def __init__(self, model_name: str = "hezarai/crnn-fa-license-plate-recognition-v2"):
        print(f"[INFO] Loading Hezar OCR model: {model_name}")
        self.model = Model.load(model_name)

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        """
        Takes a cropped plate image (BGR numpy array from cv2)
        and returns an OCRResult with transcribed text and accurate confidence.
        """
        if plate_crop is None or plate_crop.size == 0:
            return OCRResult(plate_text="", confidence=0.0)

        # 1. Convert BGR to RGB
        rgb_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

        # 2. Ensure adequate vertical height for the CNN backbone
        h, w = rgb_plate.shape[:2]
        if h < 48:
            scale = 48.0 / max(1, h)
            rgb_plate = cv2.resize(rgb_plate, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC)

        # 3. Model Inference
        try:
            results = self.model.predict(rgb_plate)
        except Exception as e:
            print(f"[WARNING] OCR prediction error: {e}")
            return OCRResult(plate_text="", confidence=0.0)

        if isinstance(results, list) and len(results) > 0:
            output = results[0]
        else:
            output = results

        # 4. Extract text string safely
        text = ""
        if isinstance(output, dict):
            text = output.get("text", "") or ""
        else:
            text = getattr(output, "text", "") or ""

        text = str(text).strip()

        # 5. Extract Confidence Score Safely (handles scalar, list, and different key names)
        raw_score = None
        if isinstance(output, dict):
            # Check all possible naming conventions across Hezar versions
            for key in ["score", "scores", "confidence", "confidences", "prob", "probs"]:
                if key in output and output[key] is not None:
                    raw_score = output[key]
                    break
        else:
            for attr in ["score", "scores", "confidence", "confidences", "prob", "probs"]:
                val = getattr(output, attr, None)
                if val is not None:
                    raw_score = val
                    break

        final_conf = 0.0
        if raw_score is not None:
            if isinstance(raw_score, (list, tuple, np.ndarray)):
                if len(raw_score) > 0:
                    # Average character-level probabilities
                    final_conf = float(sum(raw_score) / len(raw_score))
            elif isinstance(raw_score, (int, float, np.floating)):
                final_conf = float(raw_score)

        # 6. If model returned text but no confidence attribute, use heuristic fallback
        if final_conf <= 0.0 and len(text) > 0:
            final_conf = estimate_plate_confidence(text)

        # Clamp between 0.0 and 1.0
        final_conf = max(0.0, min(1.0, final_conf))

        return OCRResult(plate_text=text, confidence=final_conf)