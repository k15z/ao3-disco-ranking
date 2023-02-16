"""Test module for ao3_disco_ranking."""

from ao3_disco_ranking import __author__, __email__, __version__


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Kevin Alex Zhang"
    assert __email__ == "hello@kevz.dev"
    assert __version__ == "0.0.0"
