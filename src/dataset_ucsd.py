import os
import urllib.request
import tarfile
import sys

# Constants
DATA_DIR = "data"
URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
# Fallback mirror if the university site is slow/down
MIRROR_URL = "https://github.com/wanderine/AnomalyDetectionDataset/raw/master/UCSD_Anomaly_Dataset.tar.gz" 
FILENAME = "UCSD_Anomaly_Dataset.tar.gz"
FILE_PATH = os.path.join(DATA_DIR, FILENAME)

def download_ucsd_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Download
    if not os.path.exists(FILE_PATH):
        print(f"Downloading UCSD Dataset...")
        try:
            # Try official first
            def progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownloading... {percent}%")
                sys.stdout.flush()
            urllib.request.urlretrieve(URL, FILE_PATH, reporthook=progress)
        except Exception:
            print("\nPrimary link failed, trying mirror...")
            urllib.request.urlretrieve(MIRROR_URL, FILE_PATH)
        print("\nDownload complete.")

    # 2. Extract
    extract_path = os.path.join(DATA_DIR, "UCSD_Anomaly_Dataset.v1p2")
    if not os.path.exists(extract_path):
        print("Extracting...")
        with tarfile.open(FILE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print(f"Extracted to {extract_path}")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_ucsd_data()