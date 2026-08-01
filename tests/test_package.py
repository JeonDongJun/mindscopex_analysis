from __future__ import annotations

import subprocess
import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_root_import_does_not_eagerly_load_analysis_modules(self) -> None:
        code = (
            "import sys, mindscopex_analysis; "
            "print('mindscopex_analysis.qwen_scope' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False")

    def test_flat_public_api_still_resolves(self) -> None:
        from mindscopex_analysis import LureCase
        from mindscopex_analysis.cases import LureCase as DirectLureCase

        self.assertIs(LureCase, DirectLureCase)

    def test_all_declared_exports_resolve(self) -> None:
        import mindscopex_analysis

        for name in mindscopex_analysis.__all__:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(mindscopex_analysis, name))


if __name__ == "__main__":
    unittest.main()
