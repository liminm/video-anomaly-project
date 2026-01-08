import os
import sys
import urllib.request

# Constants
DATA_DIR = "data"
FILENAME = "mnist_test_seq.npy"
URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
FILE_PATH = os.path.join(DATA_DIR, FILENAME)


def download_data():
    # 1. Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # 2. Check if file already exists
    if os.path.exists(FILE_PATH):
        print(f"File already exists at: {FILE_PATH}")
        return

    # 3. Download the file
    print(f"Downloading Moving MNIST from {URL}...")
    try:

        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading... {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(URL, FILE_PATH, reporthook=progress)
        print(f"\nDownload complete! Saved to {FILE_PATH}")

    except Exception as e:
        print(f"\nError downloading file: {e}")


if __name__ == "__main__":
    download_data()
