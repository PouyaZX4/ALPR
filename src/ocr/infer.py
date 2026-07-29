from src.utils.data_types import PlateResult

class PlateOCR:
    def __init__(self, onnx_path: str, vocab_path: str):
        self.onnx_path = onnx_path
        self.vocab_path = vocab_path

    def read(self, plate_crop) -> PlateResult:
        """
        Runs inference on plate crop and returns OCR text result.
        """
        pass
