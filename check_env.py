"""
PhishGuard - Environment and Project Structure Verification Script.
Validates dependencies, project directories, and modular package imports.
"""

import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "pyarrow",
    "matplotlib",
    "seaborn",
    "streamlit",
    "pytest",
]

REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "reports/figures",
    "reports/results",
    "tests",
]

def check_python_version() -> bool:
    print(f"[*] Python Version: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 11):
        print("[-] WARNING: Python version is below 3.11.")
        return False
    print("[+] Python version satisfies >= 3.11 requirement.")
    return True

def check_directories() -> bool:
    print("\n[*] Checking project directory layout...")
    base_dir = Path(__file__).resolve().parent
    all_ok = True
    for d in REQUIRED_DIRECTORIES:
        target = base_dir / d
        if target.is_dir():
            print(f"  [+] Directory found: {d}")
        else:
            print(f"  [-] Directory missing: {d}")
            all_ok = False
    return all_ok

def check_dependencies() -> bool:
    print("\n[*] Checking required packages...")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [+] Package available: {pkg:<15} (version: {version})")
        except ImportError as e:
            print(f"  [-] Package MISSING:   {pkg:<15} ({e})")
            all_ok = False
    return all_ok

def check_internal_modules() -> bool:
    print("\n[*] Checking src/ package modules...")
    base_dir = Path(__file__).resolve().parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    modules = [
        "src.utils",
        "src.feature_extractor",
        "src.preprocessing",
        "src.model",
        "src.predictor",
        "src.risk_engine",
        "src.explainability",
    ]
    all_ok = True
    for m in modules:
        try:
            importlib.import_module(m)
            print(f"  [+] Module importable: {m}")
        except Exception as e:
            print(f"  [-] Module import error: {m} ({e})")
            all_ok = False
    return all_ok

def main():
    print("=" * 60)
    print("      PhishGuard: Environment & System Verification")
    print("=" * 60)

    py_ok = check_python_version()
    dir_ok = check_directories()
    dep_ok = check_dependencies()
    mod_ok = check_internal_modules()

    print("\n" + "=" * 60)
    if py_ok and dir_ok and dep_ok and mod_ok:
        print("[SUCCESS] PhishGuard environment and project structure verified!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("[FAILURE] Some checks failed. Please inspect the log above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
