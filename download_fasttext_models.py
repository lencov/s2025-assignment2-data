#!/usr/bin/env python3
"""
Script to download the fastText models for NSFW and toxic speech classification.

This script downloads the pre-trained fastText models from the Dolma project.
These models are used to classify text as NSFW or toxic speech.
"""

import os
import sys
import urllib.request
import ssl
import platform
import subprocess
from pathlib import Path

# For macOS, explicitly set SSL certificates path
if platform.system() == 'Darwin':
    # Try to install certificates first
    try:
        print("Attempting to install certificates for macOS...")
        # This line should only be run on macOS as it's specific to macOS Python installations
        result = subprocess.run(["/Applications/Python 3.x/Install Certificates.command"], 
                               shell=True, capture_output=True, text=True)
        print(f"Certificate installation result: {result.returncode}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error installing certificates: {e}")

# URLs for the models
MODEL_URLS = {
    "dolma-jigsaw-fasttext-bigrams-nsfw.bin": "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin",
    "dolma-jigsaw-fasttext-bigrams-hatespeech.bin": "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin"
}

def download_model(model_name, model_url, target_dir):
    """
    Download a model if it doesn't exist.
    
    Args:
        model_name: Name of the model file
        model_url: URL to download the model from
        target_dir: Directory to save the model in
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Full path to the model file
    model_path = os.path.join(target_dir, model_name)
    
    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    print(f"Downloading model {model_name} from {model_url}")
    print(f"This might take a while as the model files are large...")
    
    try:
        # Create a context that ignores SSL verification
        ssl_context = ssl._create_unverified_context()
        
        # Try using curl or wget if available
        if platform.system() == "Darwin" or platform.system() == "Linux":
            try:
                print("Attempting to download with curl...")
                subprocess.run(['curl', '-L', '-o', model_path, model_url], check=True)
                print(f"Successfully downloaded {model_name} to {model_path}")
                return
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"Curl failed: {e}. Trying with Python urllib...")
        
        # Fall back to urllib if curl fails
        with urllib.request.urlopen(model_url, context=ssl_context) as response, open(model_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        print(f"Successfully downloaded {model_name} to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def main():
    """
    Main function to download all models.
    """
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Target directory for the models
    target_dir = os.path.join(script_dir, "cs336_data", "models")
    
    # Download each model
    for model_name, model_url in MODEL_URLS.items():
        download_model(model_name, model_url, target_dir)
    
    print("\nAll models downloaded successfully!")
    print("You can now use the NSFW and toxic speech classification functions.")

if __name__ == "__main__":
    main() 