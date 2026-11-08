"""Verify artifact provenance, saved inference, and both Streamlit controls."""
import hashlib
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


class ModelAppTests(unittest.TestCase):
    def test_saved_artifact_and_new_feature(self):
        path = ROOT / "models" / "ice_cream_model.joblib"
        bundle = joblib.load(path)
        meta = json.loads((ROOT / "models" / "metadata.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta["model_sha256"])
        self.assertEqual(bundle["feature_names"], ["temperature_f", "is_weekend"])
        examples = pd.DataFrame({"temperature_f": [50, 80, 80, 100], "is_weekend": [0, 0, 1, 1]})
        predictions = bundle["model"].predict(examples)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertGreater(predictions[2], predictions[1])
        self.assertGreater(predictions[3], predictions[2])
        self.assertEqual(meta["train_rows"] + meta["test_rows"], 730)
        self.assertEqual(meta["data_kind"], "simulated")

    def test_interactive_app_matches_saved_model(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        initial = float(app.metric[0].value.split()[0].replace(",", ""))
        app.selectbox[0].select("Weekend").run()
        self.assertEqual(len(app.exception), 0)
        weekend = float(app.metric[0].value.split()[0].replace(",", ""))
        self.assertGreater(weekend, initial)
        app.slider[0].set_value(100).run()
        self.assertEqual(len(app.exception), 0)
        expected = joblib.load(ROOT / "models" / "ice_cream_model.joblib")["model"].predict(
            pd.DataFrame({"temperature_f": [100.0], "is_weekend": [1]})
        )[0]
        self.assertEqual(app.metric[0].value, f"{expected:,.0f} scoops")
        app.slider[0].set_value(50).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
