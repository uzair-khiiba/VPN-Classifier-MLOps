# Sample data

Synthetic data with the same columns and types as the Kaggle files (`train.parq`,
`shodan_df_hashed.csv`), made with `tests/conftest.py::make_signals`. It lets a fresh
clone run the whole pipeline (`make data SAMPLE=1 && make train SAMPLE=1`) without the 700 MB dataset.
Its labels are synthetic, so models trained on it say nothing about the real task.
