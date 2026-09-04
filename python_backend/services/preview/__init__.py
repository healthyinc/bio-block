"""Preview generators - NOT a release path.

WARNING: these generators return raw or raw-rendered pixels. The image
generator streams the uploaded bytes back unmodified, and the DICOM generator
renders the raw pixel array with any burned-in text intact. None of them
sanitizes anything.

They are no longer wired to any endpoint. Previews go through
``services.release_gate.sanitized_preview``, which produces bytes only from
verified sanitized pixels. Do not call these directly from a request handler.
"""

# Preview service package



