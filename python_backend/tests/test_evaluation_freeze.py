"""The frozen configuration, and the drift that would invalidate it.

A held-out result is evidence about one configuration. Phase 10 ran its
held-out partition, reported the number, and then extended the clinical
vocabulary - which quietly turned that number into a claim about a system that
no longer existed. Nothing caught it, because nothing was watching.

This is what watches. The freeze file records every value that can change a
release decision, and these tests fail the moment a live value drifts from it.
A drift is not a bug in these tests: it means the held-out measurement has
been spent and a new partition is needed.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FREEZE_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "evaluation_freeze.json"
)


def _freeze():
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_the_freeze_file_exists_and_names_its_phase():
    freeze = _freeze()

    assert freeze["phase"] == "phase-11"
    assert freeze["frozen_on"]


def test_corpus_version_and_size_match_the_freeze():
    from evaluations.corpus_generator import DOCUMENTS_PER_PARTITION
    from evaluations.labelled_corpus import CORPUS_VERSION, PARTITIONS

    freeze = _freeze()["corpus"]

    assert CORPUS_VERSION == freeze["version"]
    assert list(PARTITIONS) == freeze["partitions"]
    assert DOCUMENTS_PER_PARTITION == freeze["documents_per_partition"]


def test_the_held_out_partition_is_the_one_that_has_never_informed_a_decision():
    from evaluations.labelled_corpus import PARTITION_HELDOUT

    freeze = _freeze()["corpus"]

    assert PARTITION_HELDOUT == freeze["held_out_partition"]
    # Both earlier partitions are recorded as spent, with the reason. A
    # partition whose result was read is diagnostic data from then on.
    assert set(freeze["spent_partitions"]) == {"test", "heldout_v2"}


def test_model_revisions_and_checksums_match_the_freeze():
    from services.local_model_detectors import detector_specs

    freeze = _freeze()["models"]
    specs = detector_specs()

    assert set(specs) == set(freeze)
    for name, spec in specs.items():
        assert spec.revision == freeze[name]["revision"]
        assert spec.weight_sha256 == freeze[name]["weight_sha256"]
        assert spec.license == freeze[name]["license"]


def test_thresholds_match_the_freeze():
    """A threshold change invalidates the held-out result outright."""
    from services.local_model_detectors import load_locked_thresholds

    freeze = _freeze()["thresholds"]
    live = load_locked_thresholds()

    assert {n: dict(v) for n, v in live.items()} == freeze


def test_rule_versions_match_the_freeze():
    from services.clinical_vocabulary import VOCABULARY_VERSION
    from services.detection_evidence import EVIDENCE_MODEL_VERSION
    from services.modality_utility import MEASUREMENT_VERSION
    from services.transformation_manifest import MANIFEST_VERSION
    from services.utility_contract import CONTRACT_VERSION

    freeze = _freeze()["rule_versions"]

    assert VOCABULARY_VERSION == freeze["clinical_vocabulary"]
    assert EVIDENCE_MODEL_VERSION == freeze["evidence_model"]
    assert CONTRACT_VERSION == freeze["utility_contract"]
    assert MEASUREMENT_VERSION == freeze["utility_measurement"]
    assert MANIFEST_VERSION == freeze["transformation_manifest"]


def test_label_map_matches_the_freeze():
    from services.ner_phi_detector import SPACY_PHI_LABEL_MAP

    freeze = _freeze()["label_maps"]["spacy_to_internal"]

    assert dict(SPACY_PHI_LABEL_MAP) == freeze


def test_the_seed_scheme_is_recorded_so_the_corpus_can_be_rebuilt():
    freeze = _freeze()["corpus"]

    assert "partition" in freeze["random_seed_scheme"]
    assert "index" in freeze["random_seed_scheme"]


def test_targets_are_labelled_as_engineering_targets_not_legal_thresholds():
    """These numbers are what the project aimed at, and nothing more.

    A manual-review rate is an operational figure. Presenting one as a
    compliance threshold would be a claim nobody in this repository is in a
    position to make.
    """
    targets = _freeze()["engineering_targets"]

    assert targets["manual_review_rate_max"] == 0.20
    assert targets["residual_leakage_documents_max"] == 0
    assert "not legal HIPAA thresholds" in targets["_comment"]


def test_the_freeze_file_records_no_corpus_value():
    from evaluations.labelled_corpus import PARTITIONS, all_values

    serialized = FREEZE_PATH.read_text(encoding="utf-8")

    for partition in PARTITIONS:
        for value in all_values(partition):
            if len(value) < 6:
                continue
            assert value not in serialized


def test_corpus_content_matches_the_frozen_digest():
    """A version string alone does not freeze a corpus.

    Phase 10's held-out result was invalidated by a change nobody recorded,
    under a label that never moved. These digests cover each document's id,
    text and gold spans, so a silent edit fails here instead of quietly
    turning a reported number into a claim about a different corpus.
    """
    import hashlib

    from evaluations.labelled_corpus import PARTITIONS, partition_documents

    frozen = _freeze()["corpus"]["partition_sha256"]

    assert set(frozen) == set(PARTITIONS)
    for partition in PARTITIONS:
        digest = hashlib.sha256()
        for document in partition_documents(partition):
            digest.update(document.doc_id.encode())
            digest.update(b"\0")
            digest.update(document.text.encode("utf-8"))
            digest.update(b"\0")
            for span in document.spans:
                digest.update(
                    f"{span.start}:{span.end}:{span.category}\0".encode()
                )
        assert digest.hexdigest() == frozen[partition], (
            f"{partition} has changed since the configuration was frozen; "
            "the held-out result no longer describes this corpus"
        )
