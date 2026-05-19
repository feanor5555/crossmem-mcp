"""Regex-based auto-tag extraction.

Extracts ``framework``, ``framework:version``, ``source`` and ``topic`` tags
from a URL, a title and the start of the content. Designed to run in <5ms.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Pattern catalogue (compiled once)
# ---------------------------------------------------------------------------

# Versions found inside URL paths
_RE_PATH_V = re.compile(r"/v(\d+(?:\.\d+)*)", re.IGNORECASE)
_RE_PATH_DOT = re.compile(r"/(\d+\.\d+(?:\.\d+)*)/")
# Matches `@17`, `@1.2.3` after a package name (e.g. `@angular/core@17`).
_RE_AT_VERSION = re.compile(r"@(\d+(?:\.\d+)*)\b")

# Precedence of the ``_RE_NAME_*`` patterns:
#
# ``_name_version_pairs`` iterates the regexes in declaration order
# (``_RE_NAME_VERSION`` first, then ``_RE_NAME_V_VERSION``) and collects every
# match from each pattern. Both pairs are added to the tag set, so the order
# does not change *which* tags appear -- but when the same text matches both
# patterns (e.g. ``"Spring Boot 3.2"`` vs. ``"React v18"``) the first regex to
# match supplies the canonical ``framework:version`` pair, and a second match
# from the next regex is deduplicated by the surrounding ``set``. Keep
# ``_RE_NAME_VERSION`` first because it is the broader pattern (handles
# ``Name Name X.Y`` and bare ``X.Y`` versions); ``_RE_NAME_V_VERSION`` is the
# stricter ``v``-prefixed integer fallback ("Node.js v20"). Adding a new
# pattern: place it after the more specific patterns it should not shadow and
# update ``tests/core/test_tagging.py`` snapshots.

# Framework + version inline ("Spring Boot 3.2", "Django 4.2")
_RE_NAME_VERSION = re.compile(
    r"\b([A-Za-z][\w\.\-]*(?:\s+[A-Za-z][\w\.\-]*)?)\s+v?(\d+\.\d+(?:\.\d+)*)",
)

# Framework with explicit `v`-prefixed integer version ("Node.js v20", "React v18")
_RE_NAME_V_VERSION = re.compile(
    r"\b([A-Za-z][\w\.\-]*)\s+v(\d+(?:\.\d+)*)\b",
)

# Map of well-known hostnames to source tag
_SOURCE_HOSTS: dict[str, str] = {
    "developer.mozilla.org": "mdn",
    "docs.python.org": "python",
    "github.com": "github",
    "gitlab.com": "gitlab",
    "bitbucket.org": "bitbucket",
    "stackoverflow.com": "stackoverflow",
    "react.dev": "react",
    "vuejs.org": "vue",
    "angular.io": "angular",
    "angular.dev": "angular",
    "nodejs.org": "node",
    "docs.djangoproject.com": "django",
    "flask.palletsprojects.com": "flask",
    "fastapi.tiangolo.com": "fastapi",
    "spring.io": "spring",
    "kubernetes.io": "kubernetes",
}

# Extending ``_KNOWN_FRAMEWORKS`` / ``_KNOWN_TOPICS``:
#
# 1. Add a new entry below (alias -> canonical tag for frameworks; lowercase
#    string for topics). Match is word-bounded and lowercased, so write the
#    needle in lowercase including punctuation aliases (e.g. ``"node.js"``).
# 2. Update the snapshot tests in ``tests/core/test_tagging.py`` --
#    ``test_snapshot_python_asyncio`` / ``test_snapshot_github_repo`` plus any
#    new snapshot covering the added entry -- so regressions are caught.
# 3. Run ``pytest tests/core/test_tagging.py`` locally, then submit for review.

# Frameworks recognised when their name appears in URL path or title
_KNOWN_FRAMEWORKS: dict[str, str] = {
    "python": "python",
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "react": "react",
    "vue": "vue",
    "vuejs": "vue",
    "angular": "angular",
    "node": "node",
    "nodejs": "node",
    "node.js": "node",
    "next": "next",
    "nextjs": "next",
    "spring": "spring",
    "spring boot": "spring-boot",
    "spring-boot": "spring-boot",
    "kubernetes": "kubernetes",
    "docker": "docker",
    "rust": "rust",
    "go": "go",
    "golang": "go",
    "java": "java",
    "kotlin": "kotlin",
    "typescript": "typescript",
    "javascript": "javascript",
}

# Topics extracted from URL path segments / titles
_KNOWN_TOPICS: set[str] = {
    "asyncio",
    "hooks",
    "routing",
    "middleware",
    "testing",
    "logging",
    "auth",
    "authentication",
    "authorization",
    "database",
    "orm",
    "rest",
    "graphql",
    "websocket",
    "websockets",
    "streaming",
    "caching",
    "deployment",
    "performance",
    "security",
    "concurrency",
    "threading",
    "multiprocessing",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_from_host(host: str) -> str | None:
    """Return a source tag for a hostname.

    Falls back to the second-level domain (e.g. ``example-site.io`` ->
    ``example-site``) when the host is not in the allowlist.
    """
    if not host:
        return None
    host = host.lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]

    if host in _SOURCE_HOSTS:
        return _SOURCE_HOSTS[host]

    parts = host.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return host or None


def _frameworks_from_text(text: str) -> set[str]:
    """Find known framework names appearing in ``text`` (lowercased match)."""
    if not text:
        return set()
    lowered = f" {text.lower()} "
    found: set[str] = set()
    for needle, tag in _KNOWN_FRAMEWORKS.items():
        # Match as a word, but allow punctuation around (e.g. "node.js,").
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered):
            found.add(tag)
    return found


def _topics_from_text(text: str) -> set[str]:
    """Find known topic keywords inside ``text``."""
    if not text:
        return set()
    lowered = text.lower()
    return {topic for topic in _KNOWN_TOPICS if re.search(rf"\b{topic}\b", lowered)}


def _versions_from_url(url: str) -> list[str]:
    """Extract version strings from URL path patterns."""
    versions: list[str] = []
    for m in _RE_PATH_DOT.finditer(url):
        versions.append(m.group(1))
    for m in _RE_PATH_V.finditer(url):
        versions.append(m.group(1))
    for m in _RE_AT_VERSION.finditer(url):
        versions.append(m.group(1))
    return versions


def _name_version_pairs(text: str) -> list[tuple[str, str]]:
    """Return ``(framework_tag, version)`` pairs from ``Name X.Y`` style text."""
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    for regex in (_RE_NAME_VERSION, _RE_NAME_V_VERSION):
        for m in regex.finditer(text):
            raw_name = m.group(1).strip().lower()
            version = m.group(2)
            # Normalise into the known-framework tag if possible.
            tag = _KNOWN_FRAMEWORKS.get(raw_name)
            if tag is None:
                # Compose a slug if it looks like a framework name (alpha start).
                slug = re.sub(r"\s+", "-", raw_name)
                slug = re.sub(r"[^a-z0-9\-]", "", slug)
                if slug and slug[0].isalpha():
                    tag = slug
            if tag:
                pairs.append((tag, version))
    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tags(url: str, title: str = "", content: str = "") -> list[str]:
    """Return a deduplicated, sorted list of tags for the given inputs.

    Args:
        url: Source URL of the document.
        title: Document title (optional).
        content: Document content; only the start is inspected.

    Returns:
        Sorted list of tag strings. May be empty.
    """
    tags: set[str] = set()

    # Limit content scan to keep the function fast and predictable.
    content_head = content[:1024] if content else ""

    # ---- Source tag (URL host) ------------------------------------------------
    parsed = urlparse(url) if url else None
    host = parsed.hostname if parsed else None
    if host:
        src = _source_from_host(host)
        if src:
            tags.add(src)

    # ---- Frameworks from URL path, title, content head -----------------------
    path = parsed.path if parsed else ""
    haystack_for_fw = " ".join(filter(None, [path, title, content_head]))
    frameworks = _frameworks_from_text(haystack_for_fw)
    tags.update(frameworks)

    # ---- Versions from URL ---------------------------------------------------
    url_versions = _versions_from_url(url) if url else []

    # Pair URL versions with frameworks we already know about (single fw -> pair).
    if url_versions and frameworks:
        for fw in frameworks:
            for v in url_versions:
                tags.add(f"{fw}:{v}")
    elif url_versions:
        # Try to pair with a framework derived from the host (e.g. python.org).
        host_fw = _SOURCE_HOSTS.get((host or "").lower())
        if host_fw and host_fw in _KNOWN_FRAMEWORKS.values():
            for v in url_versions:
                tags.add(f"{host_fw}:{v}")

    # ---- Name+version pairs from title / content -----------------------------
    for fw_tag, version in _name_version_pairs(f"{title} {content_head}"):
        tags.add(fw_tag)
        tags.add(f"{fw_tag}:{version}")

    # ---- Topics --------------------------------------------------------------
    tags.update(_topics_from_text(f"{path} {title} {content_head}"))

    return sorted(tags)
