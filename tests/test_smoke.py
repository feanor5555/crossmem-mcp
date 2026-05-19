def test_import_crossmem():
    import crossmem

    assert crossmem is not None


def test_import_subpackages():
    from crossmem import backends, connectors, core, sources

    assert all([core, backends, sources, connectors])


def test_cli_main(capsys, monkeypatch):
    """Smoke test: ``--help`` prints help text mentioning the CLI name.

    ``main([])`` itself starts the MCP server (the way MCP clients spawn
    it), so we exercise the parser via ``--help`` instead. argparse exits
    via SystemExit on help, which is the expected behaviour.
    """
    import pytest

    from crossmem.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "crossmem" in captured.out
