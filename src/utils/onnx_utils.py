import os
import sys
import onnxruntime as ort

_GPU_CHECKED = False
_ACTIVE_PROVIDERS = ['CPUExecutionProvider']


def get_onnx_providers():
    global _GPU_CHECKED, _ACTIVE_PROVIDERS
    if _GPU_CHECKED:
        return _ACTIVE_PROVIDERS

    _GPU_CHECKED = True

    # 1. On Windows, expose PyTorch CUDA DLLs to the system loader
    if sys.platform == "win32":
        try:
            import site
            for site_dir in site.getsitepackages():
                torch_lib = os.path.join(site_dir, "torch", "lib")
                if os.path.exists(torch_lib):
                    try:
                        os.add_dll_directory(torch_lib)
                        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
                    except Exception:
                        pass
        except Exception:
            pass

    available = ort.get_available_providers()

    # 2. Check CUDA first
    if 'CUDAExecutionProvider' in available:
        try:
            test_sess = ort.InferenceSession("models/onnx/plate_detector.onnx", providers=['CUDAExecutionProvider'])
            _ACTIVE_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("\n🚀 [HARDWARE ACCELERATION] Native NVIDIA CUDA activated successfully!")
            return _ACTIVE_PROVIDERS
        except Exception as e:
            print(f"\n⚠️ [CUDA NOT COMPATIBLE] {e}")

    # 3. Check DirectML (Windows DirectX 12 GPU acceleration)
    if 'DmlExecutionProvider' in available:
        try:
            test_sess = ort.InferenceSession("models/onnx/plate_detector.onnx", providers=['DmlExecutionProvider'])
            _ACTIVE_PROVIDERS = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("\n🚀 [HARDWARE ACCELERATION] DirectML GPU activated! Running on NVIDIA GPU via DirectX 12.")
            return _ACTIVE_PROVIDERS
        except Exception as e:
            print(f"\n⚠️ [DIRECTML NOT COMPATIBLE] {e}")

    print("\n💻 [HARDWARE] Running on CPUExecutionProvider.")
    _ACTIVE_PROVIDERS = ['CPUExecutionProvider']
    return _ACTIVE_PROVIDERS