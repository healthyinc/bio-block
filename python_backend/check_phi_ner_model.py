import sys

from services.ner_phi_detector import (
    NerPhiDetectionError,
    configured_model_name,
    load_spacy_pipeline,
)


def validate_configured_model() -> str:
    """Load the configured package and verify it contains trained NER."""

    model_name = configured_model_name()
    pipeline = load_spacy_pipeline(model_name)
    if "ner" not in pipeline.pipe_names:
        raise NerPhiDetectionError("ner_model_unavailable", status_code=503)
    return model_name


def main() -> int:
    try:
        model_name = validate_configured_model()
    except NerPhiDetectionError as exc:
        print(
            f"Clinical text NER validation failed: {exc.error_code}",
            file=sys.stderr,
        )
        return 1

    print(f"Clinical text NER model ready: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
