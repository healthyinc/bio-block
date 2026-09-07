"""Per-modality utility contracts.

Privacy validation answers "is anything identifying left?". It cannot answer
"is anything useful left?". A file with every word removed passes every privacy
check ever written, and is worthless. Phase 9 shipped exactly that failure
mode: zero expected leakage with useful-text preservation of 0.214.

So each modality carries a contract stating, in versioned form:

1. what must be removed;
2. what may be consistently replaced;
3. what may be generalized;
4. what must survive unchanged for research value;
5. what is unsupported and needs a human;
6. how privacy is validated;
7. how utility is validated;
8. the minimum evidence required before automatic release.

A single "percentage preserved" number is deliberately not used across
formats: preserved text, preserved pixels and preserved statistical structure
are not the same quantity and are not comparable.

The release rule these contracts feed is three-valued, and the order matters:

* privacy cannot be established  -> block. Never overridden by utility.
* privacy passes, utility fails  -> ``manual_review_required`` with reason
  ``utility_validation_failed``. A technically safe but medically useless file
  is not reported as a successful research artifact.
* both pass                      -> release only if the policy allows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

CONTRACT_VERSION = "utility-contract-v1"

#: Reason code returned when privacy holds but the artifact is too damaged to
#: be worth releasing.
UTILITY_VALIDATION_FAILED = "utility_validation_failed"


@dataclass(frozen=True)
class UtilityContract:
    modality: str
    version: str
    must_remove: Tuple[str, ...]
    may_replace_consistently: Tuple[str, ...]
    may_generalize: Tuple[str, ...]
    must_preserve: Tuple[str, ...]
    unsupported_requires_review: Tuple[str, ...]
    privacy_validation: Tuple[str, ...]
    utility_validation: Tuple[str, ...]
    #: Metric name -> minimum value required before automatic release.
    release_evidence: Mapping[str, float] = field(default_factory=dict)
    #: True when this modality can ever release automatically at all.
    automatic_release_possible: bool = False
    notes: str = ""

    def evaluate(self, measurements: Mapping[str, float]) -> "UtilityVerdict":
        """Check measured utility against the contract's minimum evidence."""
        shortfalls: Dict[str, Dict[str, float]] = {}
        missing: list[str] = []
        for metric, minimum in self.release_evidence.items():
            if metric not in measurements:
                missing.append(metric)
                continue
            value = float(measurements[metric])
            if value < minimum:
                shortfalls[metric] = {"measured": round(value, 4), "required": minimum}
        return UtilityVerdict(
            modality=self.modality,
            contract_version=self.version,
            measurements={k: round(float(v), 4) for k, v in measurements.items()},
            shortfalls=shortfalls,
            missing_metrics=tuple(sorted(missing)),
        )


@dataclass(frozen=True)
class UtilityVerdict:
    modality: str
    contract_version: str
    measurements: Mapping[str, float]
    shortfalls: Mapping[str, Mapping[str, float]]
    missing_metrics: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """A missing metric is a failure, not a pass.

        An unmeasured contract term is indistinguishable from an unmet one,
        and treating it as satisfied is how a utility gate quietly stops
        gating anything.
        """
        return not self.shortfalls and not self.missing_metrics

    def reason_codes(self) -> Tuple[str, ...]:
        if self.passed:
            return ()
        reasons = [UTILITY_VALIDATION_FAILED]
        reasons.extend(f"utility_below_minimum_{name}" for name in sorted(self.shortfalls))
        reasons.extend(f"utility_metric_not_measured_{name}" for name in self.missing_metrics)
        return tuple(reasons)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "contract_version": self.contract_version,
            "passed": self.passed,
            "measurements": dict(self.measurements),
            "shortfalls": {k: dict(v) for k, v in self.shortfalls.items()},
            "missing_metrics": list(self.missing_metrics),
        }


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

TEXT_CONTRACT = UtilityContract(
    modality="text",
    version=CONTRACT_VERSION,
    must_remove=(
        "names of patients, relatives, household members and providers",
        "geographic subdivisions smaller than a state, and postal codes",
        "telephone, fax, email, URL, IP address",
        "social security, medical record, health plan, account, licence, "
        "certificate, device and vehicle identifiers",
        "biometric references",
        "full-face and comparable image references",
    ),
    may_replace_consistently=(
        "patient names -> PATIENT_nnn",
        "provider names -> PROVIDER_nnn",
        "facility names -> FACILITY_nnn",
        "medical record numbers -> RECORD_nnn",
        "other pinned identifier classes -> <CLASS>_nnn",
    ),
    may_generalize=(
        "ages above 89 -> 90+",
        "dates -> year only under safe_harbor_v1",
    ),
    must_preserve=(
        "diagnoses and eponymous conditions",
        "symptoms and clinical findings",
        "medications and dosages",
        "laboratory values and units",
        "measurements and vital signs",
        "procedures and imaging modalities",
        "anatomical terms and laterality",
        "negation and clinical relationships",
        "ages at or below 89",
        "sentence structure around replaced entities",
    ),
    unsupported_requires_review=(
        "age references that cannot be resolved to a number",
        "text whose residual scan still finds an identifier",
        "input that is not decodable UTF-8, or contains NUL bytes",
    ),
    privacy_validation=(
        "typed_phi_detection",
        "deterministic_redaction",
        "residual_phi_rescan",
    ),
    utility_validation=(
        "clinical_term_preservation",
        "surrogate_consistency",
        "structure_preservation",
    ),
    release_evidence={
        # Share of recorded clinical vocabulary still present after redaction.
        "clinical_term_preservation": 0.90,
        # Share of non-PHI tokens still present. Catches wholesale destruction.
        "content_token_preservation": 0.70,
    },
    automatic_release_possible=True,
    notes=(
        "The only modality that can release automatically. Both privacy and "
        "utility must pass."
    ),
)

CSV_CONTRACT = UtilityContract(
    modality="csv",
    version=CONTRACT_VERSION,
    must_remove=("direct identifier columns", "precise geography columns"),
    may_replace_consistently=("record keys within one study",),
    may_generalize=(
        "quasi-identifiers to the minimum range that satisfies k",
        "categories grouped only as far as k requires",
    ),
    must_preserve=(
        "non-identifying analytical columns",
        "row count where suppression is not required",
        "category frequencies and numeric summaries",
        "correlations between retained analytical columns",
    ),
    unsupported_requires_review=(
        "serialized output that does not match the removal plan",
        "a removed identifier value reappearing in a retained column",
    ),
    privacy_validation=(
        "safe_harbor_column_removal",
        "k_anonymity_l_diversity",
        "serialized_output_validation",
    ),
    utility_validation=(
        "row_retention",
        "cell_suppression_rate",
        "information_loss",
    ),
    release_evidence={"row_retention": 0.80, "information_loss_inverse": 0.50},
    automatic_release_possible=False,
    notes=(
        "Blocked pending the tabular release policy decision. t-closeness is "
        "explicitly out of scope."
    ),
)

PDF_CONTRACT = UtilityContract(
    modality="pdf",
    version=CONTRACT_VERSION,
    must_remove=("PHI spans in the text layer", "identifying document metadata"),
    may_replace_consistently=("names and identifiers, as for text",),
    may_generalize=("dates and ages, as for text",),
    must_preserve=(
        "non-PHI text",
        "page count",
        "clinical tables",
        "image regions outside redactions",
        "the ability to reopen and render the file",
    ),
    unsupported_requires_review=(
        "raster pages requiring OCR",
        "encrypted, macro-bearing or attachment-bearing documents",
    ),
    privacy_validation=("surface_inventory", "text_layer_residual_scan"),
    utility_validation=("non_phi_text_preservation", "page_count_preservation"),
    release_evidence={},
    automatic_release_possible=False,
    notes="Blocked: no validated PDF writer exists.",
)

WORKBOOK_CONTRACT = UtilityContract(
    modality="workbook",
    version=CONTRACT_VERSION,
    must_remove=("PHI in cells, comments, names and properties",),
    may_replace_consistently=("identifiers, as for text",),
    may_generalize=("dates and ages, as for text",),
    must_preserve=("non-PHI cells", "sheet structure", "formulas that carry no PHI"),
    unsupported_requires_review=("macros", "embedded objects", "external links"),
    privacy_validation=("surface_inventory",),
    utility_validation=("cell_preservation",),
    release_evidence={},
    automatic_release_possible=False,
    notes="Blocked: no validated workbook writer exists.",
)

DICOM_CONTRACT = UtilityContract(
    modality="dicom",
    version=CONTRACT_VERSION,
    must_remove=(
        "direct identifiers and identifying free-text metadata",
        "unknown private tags unless explicitly allowlisted",
        "overlays containing PHI",
        "burned-in PHI pixels",
        "unsafe file and directory names",
    ),
    may_replace_consistently=(
        "study, series and instance UIDs, regenerated consistently so linkage "
        "survives within the study",
        "patient and accession identifiers",
    ),
    may_generalize=("dates to year", "ages above 89 to 90+"),
    must_preserve=(
        "pixel values outside redacted regions",
        "rows, columns, bit depth, photometric interpretation, frame count",
        "modality and series relationships",
        "essential acquisition parameters",
        "spatial orientation",
        "diagnostic usability",
    ),
    unsupported_requires_review=(
        "cross-sectional head imaging, pending defacing",
        "compressed transfer syntaxes that cannot be rewritten losslessly",
        "unsupported or high-risk derived subtypes",
    ),
    privacy_validation=(
        "metadata_scrub",
        "pixel_redaction",
        "final_byte_validation",
    ),
    utility_validation=(
        "pixel_preservation_outside_redactions",
        "geometry_preservation",
        "acquisition_parameter_preservation",
    ),
    release_evidence={},
    automatic_release_possible=False,
    notes=(
        "Blocked by facial_reconstruction_not_mitigated. Derived and secondary "
        "images are quarantined for review rather than silently discarded."
    ),
)

NIFTI_CONTRACT = UtilityContract(
    modality="nifti",
    version=CONTRACT_VERSION,
    must_remove=(
        "identifying header values",
        "extensions, which can embed a whole DICOM header",
        "identifying filenames and sidecar PHI",
    ),
    may_replace_consistently=("subject identifiers within one study",),
    may_generalize=("dates to year", "ages above 89 to 90+"),
    must_preserve=(
        "voxel values outside an approved defacing mask",
        "shape, affine, qform and sform",
        "orientation and voxel spacing",
        "datatype",
        "medically relevant header values",
    ),
    unsupported_requires_review=("any volume permitting facial reconstruction",),
    privacy_validation=("header_scrub", "serialized_round_trip_validation"),
    utility_validation=("geometry_preservation", "voxel_preservation_outside_mask"),
    release_evidence={},
    automatic_release_possible=False,
    notes="Blocked pending a validated defacing tool.",
)

WSI_CONTRACT = UtilityContract(
    modality="wsi",
    version=CONTRACT_VERSION,
    must_remove=(
        "label image",
        "macro image",
        "thumbnail",
        "identifying filename",
        "XML and vendor metadata, comments",
        "visible PHI in any associated surface",
    ),
    may_replace_consistently=("slide identifiers within one study",),
    may_generalize=("dates to year",),
    must_preserve=(
        "diagnostic pyramid tiles",
        "magnification and dimensions",
        "tile structure",
        "colour profile where safe",
        "non-identifying acquisition information",
    ),
    unsupported_requires_review=("any format without a validated writer",),
    privacy_validation=("associated_surface_inventory", "tile_scan"),
    utility_validation=("diagnostic_tile_preservation", "magnification_preservation"),
    release_evidence={},
    automatic_release_possible=False,
    notes=(
        "Blocked: no validated writer. When diagnostic pixels carry no PHI "
        "they are not rewritten; only the identifying surfaces are."
    ),
)

RASTER_CONTRACT = UtilityContract(
    modality="raster",
    version=CONTRACT_VERSION,
    must_remove=("confirmed burned-in PHI regions", "EXIF, XMP and text chunks"),
    may_replace_consistently=(),
    may_generalize=(),
    must_preserve=(
        "pixels outside redacted regions",
        "confirmed non-PHI labels such as laterality, modality and scale bars",
    ),
    unsupported_requires_review=(
        "OCR text that is neither confirmed PHI nor confirmed clinical",
    ),
    privacy_validation=("region_fill", "structural_verification", "residual_ocr_scan"),
    utility_validation=("pixel_preservation_outside_redactions",),
    release_evidence={"pixel_preservation_outside_redactions": 1.0},
    automatic_release_possible=True,
    notes=(
        "Uncertain visible text routes to review rather than being blacked "
        "out or kept. No broad whitelist exists that could let a name survive."
    ),
)

CONTRACTS: Mapping[str, UtilityContract] = {
    contract.modality: contract
    for contract in (
        TEXT_CONTRACT,
        CSV_CONTRACT,
        PDF_CONTRACT,
        WORKBOOK_CONTRACT,
        DICOM_CONTRACT,
        NIFTI_CONTRACT,
        WSI_CONTRACT,
        RASTER_CONTRACT,
    )
}


def contract_for(modality: str) -> Optional[UtilityContract]:
    return CONTRACTS.get(modality)
