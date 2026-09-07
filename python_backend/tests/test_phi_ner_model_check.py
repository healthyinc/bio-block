import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_phi_ner_model  # noqa: E402
from services.ner_phi_detector import NerPhiDetectionError  # noqa: E402


def test_model_check_reports_safe_success(monkeypatch, capsys):
    monkeypatch.setattr(
        check_phi_ner_model,
        "validate_configured_model",
        lambda: "en_core_web_sm",
    )

    assert check_phi_ner_model.main() == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Clinical text NER model ready: en_core_web_sm"
    assert captured.err == ""


def test_model_check_failure_does_not_expose_internal_path(monkeypatch, capsys):
    internal_path = r"C:\private\models\phi"

    def fail_validation():
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503)

    monkeypatch.setattr(
        check_phi_ner_model,
        "validate_configured_model",
        fail_validation,
    )

    assert check_phi_ner_model.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "Clinical text NER validation failed: ner_model_unavailable"
    )
    assert internal_path not in captured.err
    assert "traceback" not in captured.err.lower()
