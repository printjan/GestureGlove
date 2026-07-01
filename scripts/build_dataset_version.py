#!/usr/bin/env python3
# scripts/build_dataset_version.py
"""
Utility script to package/archive the current dataset into a versioned release folder.
It copies the content of `data/dataset_current` into a new `data/dataset_v<index>_<current day>/`
directory without modifying the source (leaving the active dataset intact).
"""

import sys
import shutil
import re
from datetime import datetime
from pathlib import Path

# Add project src/ directory to the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

try:
    from data_fusion_project.core.paths import get_project_root
except ImportError:
    # Fallback method to get root
    def get_project_root() -> Path:
        return PROJECT_ROOT

def get_next_version_index(data_root: Path) -> int:
    """
    Scans the data directory for existing dataset_v<index>_<date> directories
    and returns the next integer index.
    """
    max_index = 0
    pattern = re.compile(r"^dataset_v(\d+)(?:_.*)?$")
    found = False
    for item in data_root.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                found = True
    
    # If no directories match, check legacy patterns too (e.g. data_v2_29-06-26) just in case
    if not found:
        legacy_pattern = re.compile(r"^data_v(\d+)(?:_.*)?$")
        for item in data_root.parent.iterdir():
            if item.is_dir():
                match = legacy_pattern.match(item.name)
                if match:
                    index = int(match.group(1))
                    if index > max_index:
                        max_index = index
                    found = True
                    
    # Default to 1 if no versions are found, otherwise increment the max found index
    return (max_index + 1) if found else 1

def main() -> int:
    base_dir = get_project_root()
    data_root = base_dir / "data"
    
    dataset_current_dir = data_root / "dataset_current"
    
    if not dataset_current_dir.exists() or not dataset_current_dir.is_dir():
        print(f"[ERROR] Active dataset directory '{dataset_current_dir}' does not exist.")
        print("Please run data collection or restore the directory structure first.")
        return 1
        
    next_idx = get_next_version_index(data_root)
    current_day = datetime.now().strftime("%d-%m-%y")
    
    target_name = f"dataset_v{next_idx}_{current_day}"
    target_dir = data_root / target_name
    
    print("─── Build Dataset Version ────────────────────────────────────")
    print(f"  Source (Current):  {dataset_current_dir.resolve()}")
    print(f"  Destination:       {target_dir.resolve()}")
    print(f"  Target Version:    v{next_idx}")
    print(f"  Current Date:      {current_day}")
    print("──────────────────────────────────────────────────────────────")
    
    if target_dir.exists():
        print(f"[ERROR] Target directory '{target_name}' already exists.")
        print("Aborting to avoid overwriting existing dataset version.")
        return 1
        
    # Copy directory ignoring system/temp files
    def ignore_patterns(path, names):
        ignored = []
        for name in names:
            if name == ".DS_Store" or name.startswith("._") or name.endswith(".tmp"):
                ignored.append(name)
        return ignored
        
    print(f"Archiving dataset to {target_name}...")
    try:
        shutil.copytree(dataset_current_dir, target_dir, ignore=ignore_patterns)
        print(f"[SUCCESS] Copied dataset version {next_idx} to {target_dir.name}")
        print("The current dataset remains intact and active.")
        return 0
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}")
        # Clean up in case copy was partial
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return 1

if __name__ == "__main__":
    sys.exit(main())
