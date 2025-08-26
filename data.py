import os
from subprocess import run
import random
import requests

# todo: don't include root files in sparse checkouts

# configuration
font = "Keposhka Regular"
lang = "eng"
count = 100

# download training data
def clone(repo: str):
    if not os.path.exists(repo):
        run([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            f"https://github.com/tesseract-ocr/{repo}.git"
        ])
        run(["git", "sparse-checkout", "init", "--cone"], cwd=repo)

def select(repo: str, dir: str):
    run(["git", "sparse-checkout", "set", dir], cwd=repo)
    run(["git", "checkout"], cwd=repo)

# read the training data
clone("langdata")
select("langdata", lang)
training = f"langdata/{lang}/{lang}.training_text"
with open(training, encoding="utf-8") as f:
    lines = [l.strip() for l in f.read().splitlines()]
random.shuffle(lines)
lines = lines[:count]

# make sure the data directory exists
out = f"tesstrain/data/{font.replace(' ', '')}-ground-truth"
os.makedirs(out, exist_ok=True)

# write ground truths
for i, line in enumerate(lines):
    # label
    name = f"{lang}_{i}"
    path = os.path.join(out, f"{name}.gt.txt")
    with open(path, "w") as f:
        f.writelines([line])
    # image
    run([
        "text2image",
        f"--font={font}",
        f"--text={path}",
        f"--outputbase={out}/{name}",
        "--max_pages=1",
        "--strip_unrenderable_words",
        "--leading=32",
        "--xsize=3600",
        "--ysize=480",
        "--char_spacing=1.0",
        "--exposure=0",
        f"--unicharset_file=langdata/{lang}/{lang}.unicharset"
    ])

# download tessdata
clone("tesseract")
select("tesseract", "tessdata")

trained = f"{lang}.traineddata"
trained_path = f"tesseract/tessdata/{trained}"
if not os.path.exists(trained_path):
    with requests.get(f"https://github.com/tesseract-ocr/tessdata_best/raw/refs/heads/main/{trained}") as res:
        res.raise_for_status()
        with open(trained_path, "wb") as f:
            f.write(res.content)