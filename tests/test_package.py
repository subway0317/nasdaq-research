import unittest

from nasdaq_research import __version__
from nasdaq_research.research import describe_project


class PackageTests(unittest.TestCase):
    def test_package_version_is_defined(self):
        self.assertTrue(__version__)

    def test_describe_project(self):
        self.assertIn("Nasdaq", describe_project())


if __name__ == "__main__":
    unittest.main()
