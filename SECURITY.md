# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅ Active |

## Reporting a Vulnerability

We take security seriously. If you discover a security issue, please **do not** open a public issue.

Instead, email us at: **security@epsionic.ai**

You should receive a response within 48 hours. If not, follow up.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (if known)

## Security Best Practices

### API Keys
- Never commit API keys to git
- Use environment variables or `.env` files (see `.env.example`)
- In production, use a secrets manager or keyring
- Rotate keys regularly

### Network
- The dashboard creates a public URL via Gradio sharing — use `--share=false` in production
- The API server listens on `0.0.0.0` by default — restrict to `127.0.0.1` if proxying via nginx
- Use HTTPS in production

### Dependencies
- Always install from PyPI (verified packages)
- Pin dependencies in production (see `pyproject.toml`)
- Run `pip audit` or `safety check` regularly
