import importlib
import os


def test_version_and_openmp_guard():
    import kunmeto

    assert kunmeto.__version__ == "0.1.0"
    # LibRaw's Fuji decoder races under OpenMP (LibRaw #845); the package pins it to 1 thread.
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_openmp_guard_forces_override_of_preexisting_value(monkeypatch):
    import kunmeto

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    importlib.reload(kunmeto)
    assert os.environ["OMP_NUM_THREADS"] == "1"
