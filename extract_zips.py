# extract_zips.py

import os, zipfile

for file in os.listdir("data"):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join("data", file), 'r') as zip_ref:
            zip_ref.extractall(os.path.join("data", file.replace(".zip", "")))
