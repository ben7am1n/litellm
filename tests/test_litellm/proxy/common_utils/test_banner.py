import io
import sys

from litellm.proxy.common_utils.banner import show_banner


def test_show_banner_does_not_fail_for_non_utf8_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    show_banner()
    stdout.flush()

    assert output.getvalue() == b"\n\n"
