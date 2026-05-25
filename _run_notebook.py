"""Run a notebook end-to-end with absolute paths (shell cwd is broken)."""
import sys
import os
from pathlib import Path

os.chdir("/u/nleobandung/vggt-attack")
sys.path.insert(0, "/u/nleobandung/vggt-attack")

import nbformat
from nbclient import NotebookClient

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name="python3",
                        resources={"metadata": {"path": str(src.parent)}})
client.execute()
nbformat.write(nb, dst)
print(f"wrote {dst}")
