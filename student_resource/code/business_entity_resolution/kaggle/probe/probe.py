"""Kaggle probe: GPU profile (spec section 5), which secret labels resolve, release asset listing.
Prints no secret values and downloads no data."""
import json, subprocess, urllib.request

try:
    import torch
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU{i}: {p.name} {p.total_memory / 2**30:.1f} GB cc {p.major}.{p.minor}")
    print("torch", torch.__version__)
except Exception as e:
    print("torch/cuda error:", e)
print(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(), "cpus")
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)

token = None
try:
    from kaggle_secrets import UserSecretsClient
    sc = UserSecretsClient()
    for label in ["GITHUB_TOKEN", "GH_TOKEN", "github_token", "GITHUB_PAT", "github", "GITHUB", "gh_token", "git_token", "GIT_TOKEN", "HF_TOKEN"]:
        try:
            v = sc.get_secret(label)
            print(f"secret {label}: available (len {len(v)})")
            if token is None and label != "HF_TOKEN":
                token = v
        except Exception as e:
            print(f"secret {label}: not available ({type(e).__name__})")
except Exception as e:
    print("kaggle_secrets error:", e)

if token:
    req = urllib.request.Request("https://api.github.com/repos/kksahu444/tri-source-entity-resolution/releases/tags/dataset-v1",
                                 headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
    try:
        rel = json.load(urllib.request.urlopen(req, timeout=30))
        for a in rel.get("assets", []):
            print("asset", a["name"], a["size"])
    except Exception as e:
        print("github release error:", e)
