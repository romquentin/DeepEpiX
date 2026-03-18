import time
import shutil
from pathlib import Path


def clear_old_cache_files_and_csv(cache_dir, max_age_minutes=120, verbose=True):
    """
    Delete files older than `max_age_minutes` from a cache directory.

    Parameters:
            cache_dir (str or Path): Path to the cache directory.
            max_age_minutes (int): Age in minutes above which files are deleted.
            verbose (bool): Whether to print info about deleted files.
    """
    now = time.time()
    cache_dir = Path(cache_dir)
    for pattern in ["*.csv", "*.png"]:
        for file in cache_dir.rglob(pattern):
            try:
                file.unlink()
                if verbose: 
                    print(f"Suppress : {file.name}")
            except Exception as e:
                if verbose: 
                    print(f"Error while deleting {file.name} : {e}")
                    
    for item in cache_dir.glob("*"):
        item_age_minutes = (now - item.stat().st_mtime) / 60
        if item_age_minutes > max_age_minutes:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
