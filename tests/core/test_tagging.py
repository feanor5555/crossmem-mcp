"""Tests for regex-based auto-tag extraction."""

from __future__ import annotations

import time

from crossmem.core.tagging import extract_tags


def test_python_docs_url_with_version() -> None:
    """Python docs URL yields python framework, version, and topic."""
    tags = extract_tags(
        "https://docs.python.org/3.12/library/asyncio.html",
        "asyncio - Asynchronous I/O",
        "",
    )
    assert "python" in tags
    assert "python:3.12" in tags
    assert "asyncio" in tags
    assert "python" in tags  # source from docs.python.org


def test_react_dev_url_with_title() -> None:
    """React docs URL yields react framework tag."""
    tags = extract_tags("https://react.dev/learn", "React Hooks Tutorial", "")
    assert "react" in tags


def test_v_path_version_pattern() -> None:
    """URL like /v17/ extracts a version-with-framework tag from the title."""
    tags = extract_tags(
        "https://example.com/v17/api",
        "Spring Boot 3.2 Reference",
        "",
    )
    # Spring Boot 3.2 -> "spring-boot:3.2" plus "spring-boot"
    assert "spring-boot:3.2" in tags
    assert "spring-boot" in tags


def test_at_version_pattern() -> None:
    """`@angular/core@17` style version is extracted."""
    tags = extract_tags(
        "https://npm.example.com/@angular/core@17",
        "Angular Core",
        "",
    )
    assert "angular" in tags
    assert "angular:17" in tags


def test_github_source_tag() -> None:
    """github.com host yields a 'github' source tag."""
    tags = extract_tags("https://github.com/foo/bar", "", "")
    assert "github" in tags


def test_mdn_source_tag() -> None:
    """developer.mozilla.org maps to 'mdn' source tag."""
    tags = extract_tags(
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "JavaScript Guide",
        "",
    )
    assert "mdn" in tags


def test_unknown_host_uses_second_level_domain() -> None:
    """Unknown hosts fall back to the second-level domain."""
    tags = extract_tags("https://docs.example-site.io/foo", "", "")
    assert "example-site" in tags


def test_django_framework_from_title() -> None:
    """Django framework is detected from a title."""
    tags = extract_tags(
        "https://docs.djangoproject.com/en/4.2/",
        "Django 4.2 documentation",
        "",
    )
    assert "django" in tags
    assert "django:4.2" in tags


def test_three_dot_path_version_pattern() -> None:
    """A `/3.12/` style path version gets paired with the framework from host."""
    tags = extract_tags("https://docs.python.org/3.12/", "", "")
    assert "python:3.12" in tags
    assert "python" in tags


def test_topics_from_url_path() -> None:
    """A topic-like path segment becomes a topic tag."""
    tags = extract_tags(
        "https://docs.python.org/3/library/asyncio.html",
        "",
        "",
    )
    assert "asyncio" in tags


def test_returns_sorted_unique_tags() -> None:
    """Returned tags are deduplicated and sorted for determinism."""
    tags = extract_tags(
        "https://docs.python.org/3.12/library/asyncio.html",
        "Python asyncio guide",
        "",
    )
    assert tags == sorted(tags)
    assert len(tags) == len(set(tags))


def test_empty_inputs_return_empty_list() -> None:
    """Calling with all-empty inputs returns an empty list."""
    assert extract_tags("", "", "") == []


def test_handles_url_without_scheme() -> None:
    """Function does not crash on a malformed URL."""
    tags = extract_tags("not a url", "Title", "")
    # Should not raise; returns possibly empty list
    assert isinstance(tags, list)


def test_node_framework_detection() -> None:
    """Node.js content is recognized."""
    tags = extract_tags("https://nodejs.org/api/", "Node.js v20 docs", "")
    assert "node" in tags
    assert "node:20" in tags


def test_performance_under_5ms() -> None:
    """extract_tags runs in <5ms on average over 100 iterations."""
    url = "https://docs.python.org/3.12/library/asyncio.html"
    title = "asyncio - Asynchronous I/O"
    content = "Python asyncio module documentation. This page describes coroutines."

    # Warmup
    extract_tags(url, title, content)

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        extract_tags(url, title, content)
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / iterations) * 1000
    assert mean_ms < 5.0, f"extract_tags too slow: {mean_ms:.3f}ms per call"


def test_snapshot_python_asyncio() -> None:
    """Snapshot: python asyncio docs produce a known set of tags."""
    tags = set(
        extract_tags(
            "https://docs.python.org/3.12/library/asyncio.html",
            "asyncio - Asynchronous I/O",
            "",
        )
    )
    expected_subset = {"python", "python:3.12", "asyncio"}
    assert expected_subset.issubset(tags)


def test_snapshot_github_repo() -> None:
    """Snapshot: github URL yields a known source tag."""
    tags = set(extract_tags("https://github.com/python/cpython", "", ""))
    assert "github" in tags
