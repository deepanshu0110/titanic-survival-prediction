import re

from tests.helpers import ROOT, run_script


def test_main_trains_and_saves_model():
    output = run_script("main.py")
    accuracies = [float(x) for x in re.findall(r"Accuracy:\s+([0-9.]+)", output)]
    assert accuracies and all(0.75 < a < 0.9 for a in accuracies)
    assert "Best model" in output
    assert (ROOT / "models" / "titanic_model.pkl").exists()
