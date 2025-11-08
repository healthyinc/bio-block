# Security Policy

We take security reports seriously. If you discover a vulnerability affecting this project, please report it privately so we can investigate and remediate before details are made public.

## Reporting a Vulnerability

- Preferred: Open a private security advisory on GitHub for this repository (see "Security" on the repository page) and select "Report a vulnerability".
- Alternatively, send an email to: security@healthyinc.org (placeholder). Please replace this address with the project maintainers' real security contact if you maintain the repository.

When reporting, please include:
- Affected component(s) and version(s)
- A concise description of the issue and impact
- Clear reproduction steps or a minimal test case (redact or anonymize any sensitive data; see below)
- Any suggested mitigations, if available

If you need to send sensitive files (for example, logs or documents that may contain personal health information), please use an encrypted channel and avoid including full patient identifiers. If emailing, consider encrypting with the PGP key listed on our project profile or using a secure file transfer link. If no secure channel is available, contact us and we will provide one.

## Scope

This policy applies to:
- Code in this repository (including javascript_backend, prototype, python_backend, and any infra-as-code related to the project)
- Public or private endpoints, API keys, or infrastructure misconfigurations that could expose data
- Leakage or unauthorized access to data stores, backups, or IPFS content referenced by the project

This policy does NOT cover:
- General support questions or feature requests — please use issues/discussions for those
- Non-security bugs that do not affect confidentiality, integrity, or availability unless they are part of an exploit chain

## Sensitive Data & Healthcare Documents

This project may handle sensitive healthcare documents and personally identifiable information (PII) or protected health information (PHI). When reporting an issue that involves such data:

- Do NOT send unredacted PHI/PII in public issues or comments.
- Provide redacted samples or synthetic test data reproducing the issue whenever possible.
- If you must share real data to reproduce an issue, contact us first and we will provide a secure, private channel and instructions for safe handling or deletion after analysis.
- We will treat reports involving PHI/PII with additional care and follow applicable data protection laws where feasible.

## Response Expectations

- Acknowledgement: We aim to acknowledge receipt within 72 hours.
- Triage: We aim to complete an initial triage and assign severity within 7 calendar days.
- Fix / Remediation: We aim to provide a remediation plan or release a fix within 30 days for high-severity issues. Timelines for critical issues may be accelerated.

These are targets and may vary based on reporter availability, complexity, and the need to coordinate with third parties.

## Public Disclosure

Please do not disclose vulnerabilities publicly until we have had a chance to investigate and release a fix (or agree on coordinated disclosure). We appreciate responsible disclosure and will coordinate timelines.

## Legal / Safe Harbor

We will not pursue legal action against good-faith security researchers who follow this policy and report vulnerabilities responsibly. If you are unsure whether your testing may be intrusive or damaging, contact us first and we'll advise on safe steps.

## Contact & Follow-up

When you report an issue, we will provide a tracking reference. If you do not receive an acknowledgement within 72 hours, or you need to escalate, please reply to your report or contact the repository administrators.

---

Note: The contact email above is a placeholder. Repository maintainers should update this file with the correct secure contact address and (optionally) a PGP key fingerprint before publishing.
