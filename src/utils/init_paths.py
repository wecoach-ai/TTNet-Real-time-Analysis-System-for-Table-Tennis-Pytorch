import os
from pathlib import Path
import sys


class InitPaths():
    def __init__(self):
        self.this_dir = os.path.dirname(__file__)

    def add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

    def add_lib(self, libs_list):
        # Add lib to PYTHONPATH
        for lib in libs_list:
            self.add_path(Path(self.this_dir) / lib)
