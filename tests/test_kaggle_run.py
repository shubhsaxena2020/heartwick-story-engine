import sys
from engine import kaggle_run as K


def test_k_decodes_non_cp1252_output(monkeypatch):
    """Kaggle CLI output with bytes cp1252 cannot decode (0x8f) must not crash _k (regression: rc=1 TypeError)."""
    monkeypatch.setattr(K, "KAGGLE", sys.executable)
    out = K._k("-c", "import sys; sys.stdout.buffer.write(b'kernel \\x8f done'); sys.stderr.buffer.write(b'\\xe2\\x9c\\x93')")
    assert out.startswith("kernel ") and "done" in out and "✓" in out


def test_k_failure_message_is_text(monkeypatch):
    monkeypatch.setattr(K, "KAGGLE", sys.executable)
    try:
        K._k("-c", "import sys; sys.stdout.buffer.write(b'\\x8f'); sys.exit(3)")
    except RuntimeError as e:
        assert "failed" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
