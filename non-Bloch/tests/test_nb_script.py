import pathlib
import runpy

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"


def test_figure_script_imports():
    """Catches broken import paths without running main()."""
    ns = runpy.run_path(str(SCRIPT), run_name="figure_script_smoke_test")
    assert callable(ns["main"])
