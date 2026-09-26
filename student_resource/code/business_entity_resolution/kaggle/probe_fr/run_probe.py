"""Kaggle CPU session: build the France wrong-ID probe (work order T1) from upload 1."""
import glob, os, sys
IN = "/kaggle/input"
find = lambda pat: sorted(glob.glob(os.path.join(IN, "**", pat), recursive=True), key=len)[0]
code = os.path.dirname(os.path.dirname(find("src/probe_fr.py")))
sys.path.insert(0, code)
from src.probe_fr import main
test_dir = os.path.dirname(find("test/test_source1.tsv"))
validator = os.path.join(os.path.dirname(os.path.dirname(test_dir)), "utils", "validate_submission.py")
main(["--upload", find("upload_1_matching_results.tsv"), "--out", "/kaggle/working/probe_fr_wrongid/matching_results.tsv",
      "--test-dir", test_dir, "--validator", validator, "--manifest", ""])
