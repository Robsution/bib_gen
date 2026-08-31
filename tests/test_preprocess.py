import json

import numpy as np

from scripts.data.preprocess import KINEMATIC_FEATURES, preprocess


def test_preprocess_preserves_documented_feature_order(tmp_path):
    raw_path = tmp_path / "families.csv"
    output_dir = tmp_path / "processed"
    raw_path.write_text(
        "1,2,3,22,10,20,30,40,1,2,3,4,0,0\n"
        "4,5,6,2112,20,40,60,80,2,4,6,8,0,0.939565\n",
        encoding="utf-8",
    )

    parent_shape, daughter_shape = preprocess(raw_path, output_dir)
    daughters = np.load(output_dir / "bib_data_daughters.npy")
    metadata = json.loads(
        (output_dir / "bib_data_scalers.json").read_text(encoding="utf-8")
    )

    assert parent_shape == (2, 3)
    assert daughter_shape == (2, 1, 11)
    assert metadata["kinematic_features"] == list(KINEMATIC_FEATURES)
    assert daughters[0, 0, 0] == 1.0
    np.testing.assert_allclose(daughters[0, 0, 1:9], -1.0)
    np.testing.assert_allclose(daughters[1, 0, 1:9], 1.0)
