"""Compatibility shim: manifests now live beside the pipeline.

Phase 11 needs a manifest for every processed artifact, not only for evaluated
ones, so the builder moved to ``services.transformation_manifest``. Evaluation
scripts import it from here as they always did.
"""

from services.transformation_manifest import (  # noqa: F401
    FORBIDDEN_KEYS,
    MANIFEST_VERSION,
    build_manifest,
    component_versions,
)

__all__ = [
    "FORBIDDEN_KEYS",
    "MANIFEST_VERSION",
    "build_manifest",
    "component_versions",
]
