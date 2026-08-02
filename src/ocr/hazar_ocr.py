# src/ocr/hazar_ocr.py

import huggingface_hub

# Patch the missing 'Repository' class in newer huggingface_hub versions
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


class PlateOCR:
    def __init__(self, model_name: str = "hezarai/crnn-fa-license-plate-recognition-v2"):
        print(f"[INFO] Loading Hezar OCR model: {model_name}")
        self.model = Model.load(model_name)

    def read(self, plate_crop) -> "OCRResult":
        """
        Takes a cropped plate image (numpy array, BGR from cv2) and returns
        an OCRResult with the transcribed text and confidence.
        """
        results = self.model.predict(plate_crop)

        if isinstance(results, list) and len(results) > 0:
            output = results[0]
        else:
            output = results

        # output may be a dict or an object -- handle both
        if isinstance(output, dict):
            text = output.get("text", "") or ""
            score = output.get("score", None)
        else:
            text = getattr(output, "text", "") or ""
            score = getattr(output, "score", None)

        if score is None:
            score = 0.0
        if isinstance(score, list):
            score = sum(score) / len(score) if score else 0.0

        return OCRResult(plate_text=text, confidence=float(score))