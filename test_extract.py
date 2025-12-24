import tarfile
import os

archive = "data/archives/images_001.tar.gz"
extract_to = "data/temp"

os.makedirs(extract_to, exist_ok=True)

with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(extract_to)

print("Extraction successful")

# show a few files
files = os.listdir(extract_to)
print("Sample files:", files[:5])
