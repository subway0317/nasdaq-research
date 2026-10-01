from nasdaq_research import __version__
from nasdaq_research.research import describe_project


def test_package_version_is_defined():
    assert __version__


def test_describe_project():
    assert "Nasdaq" in describe_project()
