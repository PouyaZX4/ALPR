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

    # Register local torch/CUDA DLLs on Windows if available
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
    print(f"\n[SYSTEM] ONNX Runtime Available Providers: {available}")

    if 'CUDAExecutionProvider' in available:
        _ACTIVE_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return _ACTIVE_PROVIDERS

    if 'DmlExecutionProvider' in available:
        _ACTIVE_PROVIDERS = ['DmlExecutionProvider', 'CPUExecutionProvider']
        return _ACTIVE_PROVIDERS

    _ACTIVE_PROVIDERS = ['CPUExecutionProvider']
    return _ACTIVE_PROVIDERS


def inspect_session_device(model_name: str, session: ort.InferenceSession):
    """
    Prints the actual provider assigned to this session.
    """
    active = session.get_providers()
    primary = active[0] if active else "None"
    
    if primary in ('CUDAExecutionProvider', 'DmlExecutionProvider', 'TensorrtExecutionProvider'):
        print(f"  [GPU ACTIVE] Model '{model_name}' running on: {primary}")
    else:
        print(f"  [CPU FALLBACK] Model '{model_name}' is using CPUExecutionProvider!")