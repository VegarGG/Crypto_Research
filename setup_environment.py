"""
Environment Setup Helper for Crypto Research Project

This script helps configure the Python environment and install all required dependencies
for running the cryptocurrency quantitative trading strategy notebooks and scripts.

Usage:
    python setup_environment.py [--install] [--check] [--arcticdb-path PATH]

Options:
    --install           Install all required packages
    --check             Check current environment and installed packages
    --arcticdb-path     Set custom ArcticDB storage path (default: ./arctic_store)
"""

import sys
import subprocess
import importlib
import argparse
from pathlib import Path

# Core dependencies with version requirements
CORE_PACKAGES = {
    'pandas': '>=1.5.0',
    'numpy': '>=1.23.0',
    'matplotlib': '>=3.6.0',
    'seaborn': '>=0.12.0',
    'scipy': '>=1.9.0',
}

# Machine Learning packages
ML_PACKAGES = {
    'scikit-learn': '>=1.2.0',
    'pycaret': '>=3.0.0',
    'lightgbm': '>=3.3.0',
    'xgboost': '>=1.7.0',
}

# Technical Analysis packages
TA_PACKAGES = {
    'ta': '>=0.10.0',  # Technical Analysis library
}

# Database and storage
DB_PACKAGES = {
    'arcticdb': '>=4.0.0',
}

# Performance optimization
PERF_PACKAGES = {
    'numba': '>=0.56.0',
}

# Jupyter environment
JUPYTER_PACKAGES = {
    'jupyter': '>=1.0.0',
    'ipykernel': '>=6.0.0',
    'ipywidgets': '>=8.0.0',
}

ALL_PACKAGES = {
    **CORE_PACKAGES,
    **ML_PACKAGES,
    **TA_PACKAGES,
    **DB_PACKAGES,
    **PERF_PACKAGES,
    **JUPYTER_PACKAGES
}


def check_package(package_name):
    """Check if a package is installed and return its version"""
    try:
        # Handle package name mapping (pip name vs import name)
        import_name = {
            'scikit-learn': 'sklearn',
            'pycaret': 'pycaret',
        }.get(package_name, package_name.replace('-', '_'))

        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def check_environment():
    """Check current environment and report package status"""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    print("\n" + "=" * 70)
    print("PACKAGE STATUS")
    print("=" * 70)

    categories = [
        ("Core Packages", CORE_PACKAGES),
        ("Machine Learning", ML_PACKAGES),
        ("Technical Analysis", TA_PACKAGES),
        ("Database", DB_PACKAGES),
        ("Performance", PERF_PACKAGES),
        ("Jupyter", JUPYTER_PACKAGES),
    ]

    all_installed = True

    for category, packages in categories:
        print(f"\n{category}:")
        print("-" * 70)

        for package, version_req in packages.items():
            installed, version = check_package(package)

            if installed:
                status = f"INSTALLED (v{version})"
                symbol = "✓"
            else:
                status = "NOT INSTALLED"
                symbol = "✗"
                all_installed = False

            print(f"  {symbol} {package:25s} {version_req:15s} {status}")

    print("\n" + "=" * 70)
    if all_installed:
        print("All packages are installed!")
        return True
    else:
        print("Some packages are missing. Run with --install to install them.")
        return False


def install_packages():
    """Install all required packages"""
    print("=" * 70)
    print("INSTALLING PACKAGES")
    print("=" * 70)

    packages_to_install = []

    for package, version_req in ALL_PACKAGES.items():
        installed, _ = check_package(package)
        if not installed:
            # Convert version requirement for pip
            if version_req.startswith('>='):
                pkg_spec = f"{package}{version_req}"
            else:
                pkg_spec = package
            packages_to_install.append(pkg_spec)

    if not packages_to_install:
        print("\nAll packages already installed!")
        return True

    print(f"\nInstalling {len(packages_to_install)} packages:")
    for pkg in packages_to_install:
        print(f"  - {pkg}")

    print("\nProceeding with installation...")

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--upgrade'
        ] + packages_to_install)

        print("\n" + "=" * 70)
        print("Installation completed successfully!")
        print("=" * 70)
        return True

    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        return False


def setup_arcticdb(custom_path=None):
    """Setup ArcticDB storage configuration"""
    print("\n" + "=" * 70)
    print("ARCTICDB CONFIGURATION")
    print("=" * 70)

    # Determine project root
    project_root = Path(__file__).parent

    # Use custom path or default
    if custom_path:
        arctic_path = Path(custom_path)
    else:
        arctic_path = project_root / 'arctic_store'

    # Create directory if it doesn't exist
    arctic_path.mkdir(parents=True, exist_ok=True)

    # Create config file
    config_file = project_root / 'config.py'

    config_content = f'''"""
Project Configuration
Auto-generated by setup_environment.py
"""

import pathlib

# Project paths
PROJECT_ROOT = pathlib.Path(__file__).parent
ARCTIC_STORE_PATH = PROJECT_ROOT / 'arctic_store'

# ArcticDB URI
ARCTIC_URI = f"lmdb://{{ARCTIC_STORE_PATH}}"

# Configuration dictionary
config = {{
    'arctic_uri': ARCTIC_URI,
    'arctic_path': ARCTIC_STORE_PATH,
    'data_path': PROJECT_ROOT / 'data',
    'models_path': PROJECT_ROOT / 'models',
    'strategy_path': PROJECT_ROOT / 'strategy',
}}
'''

    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"\nArcticDB storage path: {arctic_path}")
    print(f"Configuration saved to: {config_file}")
    print("\nArcticDB is ready to use!")


def create_requirements_file():
    """Create requirements.txt file"""
    project_root = Path(__file__).parent
    req_file = project_root / 'requirements.txt'

    print("\n" + "=" * 70)
    print("CREATING requirements.txt")
    print("=" * 70)

    with open(req_file, 'w') as f:
        f.write("# Crypto Research Project - Python Dependencies\n")
        f.write("# Generated by setup_environment.py\n\n")

        for category, packages in [
            ("# Core Data Science", CORE_PACKAGES),
            ("# Machine Learning", ML_PACKAGES),
            ("# Technical Analysis", TA_PACKAGES),
            ("# Database", DB_PACKAGES),
            ("# Performance", PERF_PACKAGES),
            ("# Jupyter", JUPYTER_PACKAGES),
        ]:
            f.write(f"\n{category}\n")
            for package, version_req in packages.items():
                f.write(f"{package}{version_req}\n")

    print(f"\nrequirements.txt created at: {req_file}")
    print("\nYou can now install using: pip install -r requirements.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Setup environment for Crypto Research Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--install', action='store_true',
                       help='Install all required packages')
    parser.add_argument('--check', action='store_true',
                       help='Check current environment')
    parser.add_argument('--arcticdb-path', type=str,
                       help='Custom ArcticDB storage path')
    parser.add_argument('--create-requirements', action='store_true',
                       help='Create requirements.txt file')

    args = parser.parse_args()

    # If no arguments, run check and setup
    if not any([args.install, args.check, args.create_requirements]):
        print("Crypto Research Project - Environment Setup")
        print("=" * 70)
        print("\nRunning environment check...")
        all_ok = check_environment()

        if not all_ok:
            print("\n" + "=" * 70)
            response = input("\nWould you like to install missing packages? (y/n): ")
            if response.lower() == 'y':
                install_packages()

        setup_arcticdb(args.arcticdb_path)
        create_requirements_file()

    else:
        if args.check:
            check_environment()

        if args.install:
            install_packages()
            setup_arcticdb(args.arcticdb_path)

        if args.create_requirements:
            create_requirements_file()

    print("\n" + "=" * 70)
    print("Setup complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
