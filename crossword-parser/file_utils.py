import sys
import os

__all__ = ( 'get_pkg_dir' )

def get_pkg_path(path=None):
    script_dir = sys.path[0]
    pkg_dir, _ = os.path.split(script_dir)

    if path is not None:
        pkg_dir = "%s/%s" % (pkg_dir, path)

    return pkg_dir

