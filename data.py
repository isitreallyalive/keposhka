import random
import os
import subprocess

# configuration
font = "Keposhka Regular"
lang = "eng"
count = 100

# read the training data
training = f"tesseract/langdata/{lang}/{lang}.training_text"
with open(training, encoding="utf-8") as f:
    lines = [l.strip() for l in f.read().splitlines()]
random.shuffle(lines)
lines = lines[:count]

# make sure the data directory exists
out = "tesseract/tesstrain/data/Keposhka-ground-truth"
if not os.path.exists(out):
    os.makedirs(out, exist_ok=True)

# write ground truths
for i, line in enumerate(lines):
    # label file
    name = f"{lang}_{i}"
    path = os.path.join(out, f"{name}.gt.txt")
    with open(path, "w") as f:
        f.writelines([line])

    # image
    subprocess.run([
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
        f"--unicharset_file=tesseract/langdata/{lang}/{lang}.unicharset"
    ])