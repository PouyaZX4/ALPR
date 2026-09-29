import os
import json
import yaml
import torch
import torch.nn as nn
from hezar.models.image2text.crnn import CRNNImage2Text, CRNNImage2TextConfig

os.makedirs("models/onnx", exist_ok=True)
output_path = "models/onnx/crnn_plate_ocr.onnx"

v2_dir = r"C:\Users\Pouya\.cache\hezar\models--hezarai--crnn-fa-license-plate-recognition-v2\snapshots\8f629aec2b5fb091e034317cb472dadd29640289"
config_path = os.path.join(v2_dir, "model_config.yaml")
weights_path = os.path.join(v2_dir, "model.pt")

print("=" * 60)
print("1. Loading exact v2 configuration...")
print("=" * 60)

with open(config_path, "r", encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)

# Save exact vocabulary
vocab = raw_config["id2label"]
with open("models/onnx/ocr_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
print(f"✅ Saved {len(vocab)} tokens to 'models/onnx/ocr_vocab.json'")

# Build config object from exact parameters
config = CRNNImage2TextConfig(
    id2label={int(k): v for k, v in vocab.items()},
    blank_id=raw_config.get("blank_id", 0),
    n_channels=raw_config.get("n_channels", 1),
    image_height=raw_config.get("image_height", 32),
    image_width=raw_config.get("image_width", 128),
    map2seq_in_dim=raw_config.get("map2seq_in_dim", 1024),
    map2seq_out_dim=raw_config.get("map2seq_out_dim", 96),
    rnn_dim=raw_config.get("rnn_dim", 256),
    reverse_output_digits=raw_config.get("reverse_output_digits", True)
)

print("[INFO] Building CRNN model architecture...")
model = CRNNImage2Text(config)

print(f"[INFO] Loading true weights from: {weights_path}")
state_dict = torch.load(weights_path, map_location="cpu")
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

cleaned_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(cleaned_dict, strict=True)  # Must be strict=True!
model.eval()
print("✅ True v2 weights loaded with 100% layer match!")

class CRNNExportWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(pixel_values=x)
        if isinstance(out, dict):
            return out.get("logits", list(out.values())[0])
        elif hasattr(out, "logits"):
            return out.logits
        return out

wrapper = CRNNExportWrapper(model)
wrapper.eval()

# Input shape: 1 Channel, Height=32, Width=384 (from image_processor_config.yaml)
dummy_input = torch.randn(1, 1, 32, 384)

print("\n[INFO] Testing forward pass with (1, 1, 32, 384)...")
with torch.no_grad():
    out = wrapper(dummy_input)
    print(f"✅ Forward pass success! Output logits shape: {out.shape}")

print(f"\n[INFO] Exporting to '{output_path}'...")
torch.onnx.export(
    wrapper,
    dummy_input,
    output_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size", 3: "width"},
        "logits": {0: "batch_size", 1: "time_steps"}
    },
    opset_version=17,
    dynamo=False
)

print("\n" + "=" * 60)
print(f"🎉 SUCCESS! Exported true v2 ONNX model: {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")
print("=" * 60)