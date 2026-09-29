import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist" / "RetailAI_v2"

print("=" * 70)
print("  RETAIL AI V2 STANDALONE EXECUTABLE BUILDER (PyInstaller)")
print("  Target: dist/RetailAI_v2/RetailAI_v2.exe")
print("=" * 70)

# 1. Run PyInstaller
cmd = [sys.executable, "-m", "PyInstaller", "--clean", "-y", "RetailAI_v2.spec"]
print(f"Running: {' '.join(cmd)}")
ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
if ret.returncode != 0:
    print(f"\n[ERROR] PyInstaller failed with return code {ret.returncode}")
    sys.exit(ret.returncode)

print("\nPyInstaller compilation succeeded! Packaging runtime asset directories...")

# 2. Ensure runtime assets exist in dist/RetailAI_v2
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

# 3. Create a starter launcher batch file in dist/RetailAI_v2
bat_content = """@echo off
title MCP Retail Analytics V2 - Standalone
cd /d "%~dp0"
echo Starting Retail AI V2 System...
RetailAI_v2.exe %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo System exited with error code %ERRORLEVEL%
    pause
)
"""
bat_path = DIST_DIR / "run_retail_ai.bat"
with open(bat_path, "w", encoding="utf-8") as f:
    f.write(bat_content)
print(f"Created launcher script: {bat_path.relative_to(PROJECT_ROOT)}")

print("\n" + "=" * 70)
print("  STANDALONE BUILD COMPLETED SUCCESSFULLY!")
print(f"  Distribution Folder: {DIST_DIR}")
print(f"  Executable:          {DIST_DIR / 'RetailAI_v2.exe'}")
print("=" * 70)
