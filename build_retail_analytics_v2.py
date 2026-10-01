import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist" / "Retail_Analytics_v2"

print("=" * 70)
print("  RETAIL ANALYTICS V2 (WINDOWED GUI) STANDALONE BUILDER")
print(f"  Target: {DIST_DIR / 'Retail_Analytics_v2.exe'}")
print("  Mode:   Windowed GUI (Zero Terminal Connection / console=False)")
print("=" * 70)

# 1. Run PyInstaller
cmd = [sys.executable, "-m", "PyInstaller", "--clean", "-y", "Retail_Analytics_v2.spec"]
print(f"Running: {' '.join(cmd)}")
ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
if ret.returncode != 0:
    print(f"\n[ERROR] PyInstaller failed with return code {ret.returncode}")
    sys.exit(ret.returncode)

print("\nPyInstaller compilation succeeded! Packaging runtime asset directories...")

# 2. Ensure runtime assets exist in dist/Retail_Analytics_v2
assets_to_copy = [
    ("config", DIST_DIR / "config"),
    ("models", DIST_DIR / "models"),
    ("mivolo_system/model", DIST_DIR / "mivolo_system" / "model"),
    ("test_videos", DIST_DIR / "test_videos"),
]

for src_rel, dst in assets_to_copy:
    src = PROJECT_ROOT / src_rel
    if src.exists():
        print(f"Copying {src_rel} -> {dst.relative_to(PROJECT_ROOT)}...")
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    else:
        print(f"Warning: Source asset directory {src} does not exist!")

# 3. Create an alias with spaces ("Retail Analytics_v2.exe")
src_exe = DIST_DIR / "Retail_Analytics_v2.exe"
dst_exe = DIST_DIR / "Retail Analytics_v2.exe"
if src_exe.exists() and not dst_exe.exists():
    shutil.copy2(src_exe, dst_exe)
    print(f"Created alias: {dst_exe.relative_to(PROJECT_ROOT)}")

# 4. Create one-click launcher batch file (launches detached, no cmd window stays open)
bat_content = """@echo off
cd /d "%~dp0"
start "" "Retail_Analytics_v2.exe"
exit
"""
bat_path = DIST_DIR / "run_retail_analytics.bat"
with open(bat_path, "w", encoding="utf-8") as f:
    f.write(bat_content)
print(f"Created launcher script: {bat_path.relative_to(PROJECT_ROOT)}")

print("\n" + "=" * 70)
print("  STANDALONE WINDOWED BUILD COMPLETED SUCCESSFULLY!")
print(f"  Distribution Folder: {DIST_DIR}")
print(f"  Executable:          {DIST_DIR / 'Retail_Analytics_v2.exe'}")
print(f"  Alias Executable:    {DIST_DIR / 'Retail Analytics_v2.exe'}")
print("=" * 70)
