import pathlib
import tqdm

PATH = pathlib.Path(".dvc/cache/files/md5")
# touch every file in PATH and its subdirectories
# because NFS file system and permissions issue
for p in tqdm.tqdm(PATH.glob("**/*")):
    p.touch()
