#!/usr/bin/env python3
"""
Script to download the fastText language identification model.
This model is required for the language identification functionality.
"""

import os
import sys
import urllib.request
import hashlib

# Model information
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_FILE = "lid.176.bin"
EXPECTED_MD5 = "7e69ec3c33a763c62ddb94c249ae698c"  # MD5 hash of the file

def download_model():
    """Download the fastText language identification model."""
    if os.path.exists(MODEL_FILE):
        # Check if file is valid by comparing MD5 hash
        md5 = hashlib.md5()
        with open(MODEL_FILE, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        if md5.hexdigest() == EXPECTED_MD5:
            print(f"Model already exists at {os.path.abspath(MODEL_FILE)}")
            return
        else:
            print("Existing model file appears corrupted. Re-downloading...")
            
    print(f"Downloading fastText language identification model to {os.path.abspath(MODEL_FILE)}...")
    print("This might take a while (the file is around 131 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)
        
    # Verify the downloaded file
    md5 = hashlib.md5()
    with open(MODEL_FILE, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    if md5.hexdigest() != EXPECTED_MD5:
        print("Warning: Downloaded file does not match expected MD5 hash.")
        print("The model may be corrupted or has been updated.")
    else:
        print("Model file verified successfully.")

if __name__ == "__main__":
    download_model() 