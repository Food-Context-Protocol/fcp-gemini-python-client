"""URL validation to prevent SSRF attacks.

CRITICAL: This module prevents Server-Side Request Forgery (SSRF) attacks
by validating URLs before fetching them server-side.

Vulnerabilities prevented:
- SSRF (CVSS 9.1): Fetching internal/private URLs
- Arbitrary file read (CVSS 9.8): file:// protocol
- Cloud metadata access: 169.254.169.254, metadata.google.internal
"""

import ipaddress
import os
import re
from urllib.parse import urlparse

import httpx


class ImageURLError(Exception):
    """Raised when an image URL fails validation."""

    pass


def _is_production() -> bool:
    """Check if running in production environment."""
    env = os.environ.get("ENVIRONMENT", "").lower()
    # Also check for common production indicators
    return env in {"production", "prod"} or os.environ.get("K_SERVICE") is not None


# Allowed URL schemes - HTTPS only in production, HTTP allowed for local dev
ALLOWED_SCHEMES = {"https"} if _is_production() else {"https", "http"}

# Allowed image domains (whitelist approach)
_BASE_ALLOWED_DOMAINS = {
    # Google Cloud Storage
    "firebasestorage.googleapis.com",
    "storage.googleapis.com",
    "storage.cloud.google.com",
    # Common CDNs
    "res.cloudinary.com",
    "images.unsplash.com",
    # Wikimedia/Wikipedia
    "upload.wikimedia.org",
    "commons.wikimedia.org",
    # Flickr
    "live.staticflickr.com",
    # Pixabay
    "cdn.pixabay.com",
}

# Add localhost only in development
ALLOWED_DOMAINS = _BASE_ALLOWED_DOMAINS if _is_production() else _BASE_ALLOWED_DOMAINS | {"localhost", "127.0.0.1"}

# Blocked IP ranges (internal/private networks)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::1/128"),
]

# Blocked hostnames (cloud metadata endpoints)
BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.goog",
    "169.254.169.254",
    "metadata",
    "instance-data",
}

# Allowed image content types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/heic",
    "image/heif",
}


def _is_ip_blocked(hostname: str) -> bool:
    """Check if a hostname is a blocked IP address."""
    try:
        ip = ipaddress.ip_address(hostname)
        return any(ip in blocked_range for blocked_range in BLOCKED_IP_RANGES)
    except ValueError:
        return False


def validate_image_url(
    url: str,
    allow_any_domain: bool = False,
    additional_domains: set[str] | None = None,
) -> str:
    """Validate an image URL to prevent SSRF attacks."""
    if not url or not isinstance(url, str):
        raise ImageURLError("URL is required")

    url = url.strip()

    # Block dangerous schemes
    if url.lower().startswith("file://"):
        raise ImageURLError("file:// URLs are not allowed")
    if url.lower().startswith("data:"):
        raise ImageURLError("data: URLs are not allowed")
    if url.lower().startswith("ftp://"):
        raise ImageURLError("ftp:// URLs are not allowed")

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ImageURLError("Invalid URL format") from e

    # Validate scheme
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        if _is_production() and parsed.scheme.lower() == "http":
            raise ImageURLError("HTTP URLs are not allowed in production. Use HTTPS.")
        raise ImageURLError(f"URL scheme must be http or https, got: {parsed.scheme}")

    # Get hostname
    hostname = parsed.hostname
    if not hostname:
        raise ImageURLError("URL must have a hostname")

    hostname_lower = hostname.lower()

    # Block known dangerous hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        raise ImageURLError(f"Access to {hostname} is not allowed")

    # Block private IP addresses
    if _is_ip_blocked(hostname):
        raise ImageURLError("Access to private/internal IP addresses is not allowed")

    # Check domain whitelist
    if not allow_any_domain:
        allowed = ALLOWED_DOMAINS.copy()
        if additional_domains:
            allowed.update(additional_domains)

        domain_allowed = any(
            hostname_lower == allowed_domain or hostname_lower.endswith(f".{allowed_domain}")
            for allowed_domain in allowed
        )
        if not domain_allowed:
            raise ImageURLError(
                f"Domain {hostname} is not in the allowed list. Allowed domains: {', '.join(sorted(allowed))}"
            )

    # Block URLs with suspicious patterns
    if re.search(r"[@]", parsed.netloc):
        raise ImageURLError("URLs with credentials are not allowed")

    if port := parsed.port:
        allowed_ports = {80, 443} if _is_production() else {80, 443, 8080, 8000, 3000}
        if port not in allowed_ports:
            raise ImageURLError(f"Non-standard port {port} is not allowed")

    return url


def validate_content_type(content_type: str | None) -> bool:
    """Validate that a Content-Type header is an allowed image type."""
    if not content_type:
        return False

    base_type = content_type.split(";")[0].strip().lower()
    return base_type in ALLOWED_CONTENT_TYPES


async def verify_url_reachability(url: str, timeout: float = 5.0) -> bool:
    """Verify that a URL is reachable and returns a valid image."""
    try:
        validate_image_url(url, allow_any_domain=True)
    except ImageURLError:
        return False

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.head(url, follow_redirects=True)

            if response.status_code != 200:
                if response.status_code in [403, 404, 405]:
                    try:
                        async with client.stream("GET", url) as stream_response:
                            if stream_response.status_code == 200:
                                return validate_content_type(stream_response.headers.get("Content-Type"))
                    except Exception:
                        pass
                return False

            return validate_content_type(response.headers.get("Content-Type"))

    except Exception:
        return False
