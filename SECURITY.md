# Security Policy

This repository follows a responsible disclosure policy for reporting security vulnerabilities.

**Reporting a Vulnerability**

- Preferred: Open a private security advisory on GitHub for this repository (see the "Security" tab on the repository page) and select "Report a vulnerability".
- Alternatively, send an email to `security@healthyinc.org` (placeholder). If you are a maintainer, replace this placeholder with the real security contact address before merging.

Do NOT report security issues in public issue trackers, pull requests, or discussion threads.

**What to include**

- A clear description of the vulnerability and affected components.
- Steps to reproduce or a minimal proof-of-concept (if safe to share).
- Impact assessment describing what data or systems are affected.
- Any suggested mitigations or fixes, if you have them.

**Scope**

The following components are in-scope for this policy unless specifically noted otherwise:

- `javascript_backend/` (Node.js API and controllers)
- `python_backend/` (Python API)
- `prototype/` (React frontend and smart contract code)
- Any CI/CD workflows, configuration, or deployment routes defined in this repository

If you are unsure whether something is in scope, report it and the maintainers will triage.

**Handling PHI/PII**

If the vulnerability involves protected health information (PHI) or personally identifiable information (PII):

- Avoid sharing actual PHI/PII in public communications. Use synthetic or redacted samples in your report.
- Clearly mark any sensitive data in attachments or PoCs and provide secure access if needed.

**Triage and Response Expectations**

- Acknowledgement: Maintainers will acknowledge receipt within 72 hours.
- Initial triage: We'll attempt to triage and classify severity within 7 days.
- Remediation: Time to fix depends on severity and complexity. Critical issues will be prioritized.
- Disclosure: Public disclosure will be coordinated with the reporter; we will not disclose before a fix is available unless required by law.

**Safe Harbor**

By reporting a vulnerability in good faith, you agree to follow responsible disclosure practices and avoid any actions that would increase risk to users or systems. Do not exploit the vulnerability beyond what is necessary to demonstrate it.

**Credits**

We appreciate responsible disclosure. Reporters who follow this policy may be credited in release notes or an acknowledgements file with their permission.
