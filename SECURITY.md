# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Please report security vulnerabilities by emailing **security@fcp.dev**.

You will receive an acknowledgment within 48 hours. We will work with you to understand and address the issue before any public disclosure.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Measures

- All secrets are managed via Google Cloud Secret Manager
- Authentication is required for write operations
- Input sanitization includes prompt injection prevention
- SSRF protection is enabled on all outbound requests
- Rate limiting is enforced on all endpoints
- Dependencies are monitored via Dependabot

## Scope

This policy applies to all repositories under the [Food-Context-Protocol](https://github.com/Food-Context-Protocol) organization.
