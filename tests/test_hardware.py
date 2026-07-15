from types import SimpleNamespace

from histreggui.hardware import configure_registration_device, detect_cuda


class FakeCUDA:
    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def device_count(self):
        return 1 if self.available else 0

    def get_device_name(self, index):
        return "Test GPU"

    def synchronize(self):
        return None


class FakeTorch:
    def __init__(self, cuda_version="12.8", available=True):
        self.version = SimpleNamespace(cuda=cuda_version)
        self.cuda = FakeCUDA(available=available)

    def empty(self, *_args, **_kwargs):
        return object()


def test_detect_cuda_available():
    info = detect_cuda(FakeTorch(), probe=True)
    assert info.available is True
    assert info.device_names == ("Test GPU",)


def test_detect_cpu_only_build():
    info = detect_cuda(FakeTorch(cuda_version=None), probe=False)
    assert info.available is False
    assert info.compiled_with_cuda is False


def test_configure_registration_device_cpu():
    params = {
        "device": "cuda:0",
        "initial": {"cuda": True, "device": "cuda:0"},
        "items": ["cuda:1", {"device": "cuda"}],
    }
    result = configure_registration_device(params, "cpu")
    assert result["device"] == "cpu"
    assert result["initial"]["cuda"] is False
    assert result["initial"]["device"] == "cpu"
    assert result["items"][0] == "cpu"
    assert result["items"][1]["device"] == "cpu"


def test_configure_registration_device_cuda():
    params = {"device": "cpu", "nested": {"cuda": False, "device": "cpu"}}
    result = configure_registration_device(params, "cuda:0")
    assert result["device"] == "cuda:0"
    assert result["nested"]["cuda"] is True
    assert result["nested"]["device"] == "cuda:0"
