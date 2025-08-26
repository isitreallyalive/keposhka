import os
from subprocess import run
import random

# configuration
font = "Keposhka Regular"
lang = "eng"
count = 100

# download training data
if not os.path.exists("langdata"):
    run([
        "git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/tesseract-ocr/langdata.git"
    ])
    run(["git", "sparse-checkout", "init", "--cone"], cwd="langdata")
run(["git", "sparse-checkout", "set", lang], cwd="langdata")
run(["git", "checkout"], cwd="langdata")

# read the training data
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