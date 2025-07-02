#!/usr/bin/env python3
import os
import sys
import shutil
import stat
import platform
from pathlib import Path

def main():
    # 1. Defining paths
    project_root = Path(__file__).resolve().parents[2]
    src_file = project_root / "kaggle.json"
    if not src_file.exists():
        print(f"File {src_file} was not found. Put kaggle.json in the root of the repo.", file=sys.stderr)
        sys.exit(1)

    home = Path.home()
    if platform.system() == "Windows":
        dest_dir = home / ".kaggle"
    else:
        # Linux / macOS
        dest_dir = home / ".kaggle"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / "kaggle.json"

    # 2. Copy
    shutil.copy2(src_file, dest_file)
    print(f"Copied: {src_file} → {dest_file}")

    # 3. Set permissions
    if platform.system() != "Windows":
        # only on Unix-like systems chmod 600
        dest_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        print(f"Permissions are set: chmod 600 {dest_file}")
    else:
        # on Windows you can set the “read-only” attribute if you need to
        # dest_file.chmod(stat.S_IREAD)
        print("On Windows, you do not need to change permissions (or set them manually).")

    print("kaggle.json was successfully installed.")

if __name__ == "__main__":
    main()
