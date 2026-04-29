import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
"""
Download Kaggle dataset and extract CSVs for ingestion.

This script uses the `kaggle` Python package. It will read `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment (or `.env`) and, if present, write a temporary `~/.kaggle/kaggle.json` file for authentication.

Usage:
  python scripts/download_kaggle.py --dataset iveeaten3223times/massive-yahoo-finance-dataset --out data/kaggle_yahoo

Notes:
- You must have `kaggle` installed (it's in `requirements.txt`).
- Alternatively, you can manually download the dataset from the Kaggle web UI and place CSVs into `data/kaggle_yahoo/`.
"""
import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from backend.scripts.script_logger import logger

load_dotenv()

KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')


def write_kaggle_json(username: str, key: str):
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    creds_path = kaggle_dir / 'kaggle.json'
    creds = {'username': username, 'key': key}
    with open(creds_path, 'w', encoding='utf-8') as f:
        json.dump(creds, f)
    try:
        # attempt to set restrictive permissions on Unix-like systems
        os.chmod(creds_path, 0o600)
    except Exception as e:
        logger.warning("Failed to set permissions on Kaggle credentials file %s: %s", creds_path, e)
    logger.info('Wrote Kaggle credentials to %s', creds_path)


def download_dataset(dataset: str, out_dir: str):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        logger.error('kaggle package not installed or import failed: %s', e)
        logger.error('Install requirements with `pip install -r requirements.txt`')
        return
    api = KaggleApi()
    api.authenticate()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading dataset %s to %s (this may take a while)', dataset, out_path)
    api.dataset_download_files(dataset, path=str(out_path), unzip=True, quiet=False)
    logger.info('Download completed')


def main():
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='iveeaten3223times/massive-yahoo-finance-dataset')
    parser.add_argument('--out', required=False, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kaggle_yahoo')))
    args = parser.parse_args()

    if KAGGLE_USERNAME and KAGGLE_KEY:
        write_kaggle_json(KAGGLE_USERNAME, KAGGLE_KEY)
    else:
        logger.warning('KAGGLE_USERNAME and KAGGLE_KEY not set in environment. If you have a kaggle.json in ~/.kaggle, the script will use it. Otherwise, set credentials in .env or download manually from the Kaggle web UI.')

    download_dataset(args.dataset, args.out)


if __name__ == '__main__':
    main()
