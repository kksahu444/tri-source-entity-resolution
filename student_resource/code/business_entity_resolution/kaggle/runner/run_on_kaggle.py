"""Run the full pipeline on a Kaggle session (GPU T4 x2 or TPU VM), from the private datasets
aditya103856/amlc26-er-data (organisers' zip) and aditya103856/amlc26-er-code (this code).

Outputs in /kaggle/working: output/{matching_results,candidate_pairs}.tsv (validated), runs/ (run log,
stage logs), ckpt/ (models and small artifacts). Tree models train on the GPU (XGBoost CUDA) when one is
present; CPU stages use every core the session has.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

IN, TMP, W = "/kaggle/input", "/kaggle/tmp", "/kaggle/working"
CONFIG = os.environ.get("ER_CONFIG", "config_v7.yaml")
STAGE = os.environ.get("ER_STAGE", "all")


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def find(pattern):
    hits = glob.glob(os.path.join(IN, "**", pattern), recursive=True)
    return sorted(hits, key=len)[0] if hits else None


def main():
    t0 = time.time()
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(f"{W}/runs", exist_ok=True)
    print("cpus:", os.cpu_count(), flush=True)
    sh("free -g", check=False)
    gpu = shutil.which("nvidia-smi") is not None and sh("nvidia-smi --query-gpu=name,memory.total --format=csv",
                                                           check=False).returncode == 0
    # packages the Kaggle image may lack (pure wheels; numpy/pandas/pyarrow/sklearn/lightgbm/xgboost come with it)
    sh(f"{sys.executable} -m pip install -q rapidfuzz sparse_dot_topn anyascii pyyaml sentencepiece protobuf", check=False)
    try:
        import numba  # noqa: F401
    except ImportError:
        sh(f"{sys.executable} -m pip install -q numba", check=False)

    # code
    rp = find("src/run_pipeline.py")
    if rp is None:
        z = find("code*.zip")
        zipfile.ZipFile(z).extractall(f"{TMP}/code_zip")
        rp = sorted(glob.glob(f"{TMP}/code_zip/**/src/run_pipeline.py", recursive=True), key=len)[0]
    code = f"{TMP}/code"
    shutil.copytree(os.path.dirname(os.path.dirname(rp)), code, dirs_exist_ok=True)

    # data: auto-extracted by Kaggle, or the zip itself
    ts1 = find("test/test_source1.tsv")
    if ts1 is None:
        z = find("*student_resource*.zip")
        print("extracting", z, flush=True)
        zipfile.ZipFile(z).extractall(f"{TMP}/sr")
        ts1 = sorted(glob.glob(f"{TMP}/sr/**/test/test_source1.tsv", recursive=True), key=len)[0]
    dataset = os.path.dirname(os.path.dirname(ts1))
    validator = os.path.join(os.path.dirname(dataset), "utils", "validate_submission.py")
    print("dataset:", dataset, "validator present:", os.path.exists(validator), flush=True)

    # config
    import yaml
    cfg = yaml.safe_load(open(f"{code}/{CONFIG}", encoding="utf-8"))
    cfg["paths"] = {"dataset": dataset, "artifacts": f"{TMP}/artifacts", "runs": f"{W}/runs",
                    "output": f"{W}/output", "submissions": f"{W}/submissions"}
    cfg["normalize"]["workers"] = os.cpu_count()
    cfg["model"] = {"backend": "xgboost_gpu" if gpu else "lightgbm"}
    if not gpu and (cfg.get("cross_encoder") or {}).get("enabled"):
        cfg["cross_encoder"]["enabled"] = False            # CPU session: no cross-encoder (the GPU kernel runs it)
        print("no GPU: cross-encoder disabled", flush=True)
    yaml.safe_dump(cfg, open(f"{code}/config_kaggle.yaml", "w", encoding="utf-8"), sort_keys=False)
    print("backend:", cfg["model"]["backend"], flush=True)

    # resume: artifacts from an earlier session's ckpt, if attached as input
    prev = find("ckpt/artifacts")
    if prev:
        shutil.copytree(prev, f"{TMP}/artifacts", dirs_exist_ok=True)
        print("resumed artifacts from", prev, flush=True)

    env = {**os.environ, "ER_WORKERS": str(max(1, os.cpu_count() - 1)), "PYTHONUNBUFFERED": "1"}
    stages = STAGE.split(",")
    rc = 0
    for st in stages:
        split = ""
        if ":" in st:
            st, split = st.split(":")
        cmd = f"cd {code} && {sys.executable} -u -m src.run_pipeline --config config_kaggle.yaml --stage {st}"
        cmd += f" --split {split}" if split else ""
        cmd += f" 2>&1 | tee -a {W}/runs/pipeline.log; exit ${{PIPESTATUS[0]}}"
        rc = subprocess.run(["bash", "-c", cmd], env=env).returncode
        print(f"stage {st}:{split} exit {rc} after {(time.time() - t0) / 60:.0f} min", flush=True)
        if rc != 0:
            break

    # work-order analyses on this run's artifacts (T4 error budget, T2 transfer checks); failures don't
    # affect the submission files already written
    tag = os.environ.get("ER_TAG", "kaggle-" + time.strftime("%m%d-%H%M"))
    if rc == 0:
        for mod in ("src.error_analysis", "src.transfer"):
            cmd = f"cd {code} && {sys.executable} -u -m {mod} --config config_kaggle.yaml --tag {tag} 2>&1 | tee -a {W}/runs/analysis.log"
            subprocess.run(["bash", "-c", cmd], env=env)
        # acceptance check (runs/decoy_shift.md): near-house decoy links vs train truth; result in runs/acceptance.jsonl
        cmd = (f"cd {code} && {sys.executable} -u -m src.acceptance --config config_kaggle.yaml "
               f"--file {W}/output/matching_results.tsv 2>&1 | tee -a {W}/runs/analysis.log")
        subprocess.run(["bash", "-c", cmd], env=env)

    # checkpoint small artifacts (models, vocabularies, decisions) for resumes and download
    ck = f"{W}/ckpt/artifacts"
    os.makedirs(ck, exist_ok=True)
    for pat in ("vocab.json", "dev_universe.parquet", "decision.pkl", "translit_dict.json", "feat/*.pkl",
                "ce/summary.json", "ce/*_logits.parquet", "ce/FAILED", "ce/*_f?.json",
                "pred/test_France.parquet", "pred/test_US.parquet", "pred/test_India.parquet",
                "pred/test_selected.parquet"):   # final test probabilities, for decision-layer work offline
        for f in glob.glob(f"{TMP}/artifacts/{pat}"):
            dst = os.path.join(ck, os.path.relpath(f, f"{TMP}/artifacts"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(f, dst)
    print(f"done, exit {rc}, {(time.time() - t0) / 60:.0f} min", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
