# Licences

## Models

Every learned component is trained here on the provided training data. Tree models: LightGBM (MIT) or,
on a GPU, XGBoost (Apache-2.0). The optional cross-encoder fine-tunes a public pretrained backbone.

| Component | Licence | Parameters | Status |
|---|---|---|---|
| Pruner, stage A, stage B, count model | LightGBM MIT / XGBoost Apache-2.0 (libraries) | tree ensembles trained here | in the pipeline |
| microsoft/mdeberta-v3-base | MIT | 278M | cross-encoder bake-off candidate (stage `ce`), used only if it wins the bake-off and the cross-encoder variant wins on dev |
| FacebookAI/xlm-roberta-base | MIT | 278M | cross-encoder bake-off candidate (stage `ce`), same condition |
| intfloat/multilingual-e5-small | MIT | 118M | optional dense channel C4 (Kaggle kernel `kaggle/dense`), not in the pipeline |
| BAAI/bge-m3 | MIT | 568M | optional dense channel C4, not in the pipeline |

No model outside MIT / Apache-2.0, and none above 8B parameters, is used anywhere.

## Python packages (`pip-licenses` output for requirements.txt)

| Name            | Version | License                                         | URL                                         |
|-----------------|---------|-------------------------------------------------|---------------------------------------------|
| PyYAML          | 6.0.3   | MIT License                                     | https://pyyaml.org/                         |
| RapidFuzz       | 3.14.6  | MIT                                             | https://github.com/rapidfuzz/RapidFuzz      |
| anyascii        | 0.3.3   | ISC License (ISCL)                              | https://github.com/anyascii/anyascii        |
| joblib          | 1.5.2   | BSD License                                     | https://joblib.readthedocs.io               |
| lightgbm        | 4.7.0   | MIT                                             | https://github.com/lightgbm-org/LightGBM    |
| llvmlite        | 0.49.0  | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | http://llvmlite.readthedocs.io              |
| numba           | 0.67.0  | BSD License                                     | https://numba.pydata.org                    |
| numpy           | 2.2.6   | BSD License                                     | https://numpy.org                           |
| pandas          | 2.3.3   | BSD License                                     | https://pandas.pydata.org                   |
| pyarrow         | 23.0.1  | Apache-2.0                                      | https://arrow.apache.org/                   |
| pytest          | 8.2.2   | MIT License                                     | https://docs.pytest.org/en/latest/          |
| scikit-learn    | 1.7.2   | BSD-3-Clause                                    | https://scikit-learn.org                    |
| scipy           | 1.16.3  | BSD License                                     | https://scipy.org/                          |
| sparse-dot-topn | 1.2.0   | Apache Software License                         | https://github.com/ing-bank/sparse_dot_topn |
| xgboost         | 3.4.1   | Apache-2.0                                      | https://github.com/dmlc/xgboost             |

GPU-only extras (`requirements-gpu.txt`, cross-encoder stage): torch (BSD-3-Clause), transformers
(Apache-2.0), sentencepiece (Apache-2.0), protobuf (BSD-3-Clause), in the versions of the Kaggle GPU image.
