"""
PhishGuard Dataset Downloader.

Downloads the benchmark PhiUSIIL Phishing URL Dataset from the official
UCI Machine Learning Repository into data/raw/ and uncompresses it safely.
"""

import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_RAW_DIR, setup_logger

logger = setup_logger("DatasetDownloader")

UCI_PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"

def download_and_extract(url: str = UCI_PHIUSIIL_URL, dest_dir: Path = DATA_RAW_DIR) -> Path:
    """Downloads the PhiUSIIL zip from UCI ML Repository and extracts contents.

    Args:
        url: Direct download URL for the dataset archive.
        dest_dir: Target directory where data should be unpacked.

    Returns:
        Path to the primary extracted CSV file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "phiusiil_phishing_url_dataset.zip"

    # Check if CSV already exists to avoid redundant download
    existing_csvs = list(dest_dir.glob("*.csv"))
    if existing_csvs:
        logger.info(f"Dataset CSV already present: {existing_csvs[0].name}")
        return existing_csvs[0]

    logger.info(f"Initiating download from official repository: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "PhishGuard-Academic-Research/1.0"})

    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        chunk_size = 1024 * 1024  # 1 MB chunks
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            print(f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)

    print()
    logger.info("Download completed. Unpacking archive...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    extracted_files = list(dest_dir.glob("*"))
    logger.info(f"Files in {dest_dir}: {[f.name for f in extracted_files]}")

    csv_candidates = list(dest_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file discovered after unzipping {zip_path}")

    primary_csv = csv_candidates[0]
    logger.info(f"Primary dataset ready at: {primary_csv} ({primary_csv.stat().st_size / (1024 * 1024):.2f} MB)")
    return primary_csv

if __name__ == "__main__":
    download_and_extract()
