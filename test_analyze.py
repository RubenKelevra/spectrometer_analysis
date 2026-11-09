import ast
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("analyze.py")
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
reader = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "read_spectral_data"
)
namespace = {}
exec(  # noqa: S102
    compile(ast.Module(body=[reader], type_ignores=[]), str(SOURCE), "exec"), namespace
)
read_spectral_data = namespace["read_spectral_data"]


class SpectralDataParsingTests(unittest.TestCase):
    def parse(self, contents: str) -> dict[float, float]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.txt"
            path.write_text(contents, encoding="utf-8")
            return read_spectral_data(str(path))

    def test_comma_separated_measurements(self) -> None:
        data = self.parse("380.0,0.0\n381.0,0.125\n382.0,0.25\n")
        self.assertEqual(data, {380.0: 0.0, 381.0: 0.125, 382.0: 0.25})

    def test_whitespace_separated_measurements(self) -> None:
        data = self.parse("nm intensity\n--- ---\n380.0 0.0\n381.0 0.125\n")
        self.assertEqual(data, {380.0: 0.0, 381.0: 0.125})

    def test_missing_measurements_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No spectral data"):
            self.parse("nm intensity\n")


if __name__ == "__main__":
    unittest.main()
