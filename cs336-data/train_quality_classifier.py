#!/usr/bin/env python3
"""
Script to train a fastText quality classifier using:
- Wikipedia reference URLs (high quality)
- Common Crawl samples (low quality)

This trains a classifier to distinguish between high-quality and low-quality web content.
"""

import os
import re
import random
import argparse
import gzip
import sys
import urllib.request
import fasttext
from pathlib import Path
from typing import List, Tuple
import shutil
import tempfile

# Try to import warcio with error handling
try:
    from warcio.archiveiterator import ArchiveIterator
except ImportError:
    print("Error: The 'warcio' module is not installed.")
    print("Please install it using: pip install warcio")
    print("Or install all requirements: pip install -r requirements.txt")
    sys.exit(1)

# Import our custom modules
try:
    from cs336_data.data import extract_text_from_html_bytes, identify_language, gopher_quality_filter
except ImportError as e:
    print(f"Error importing from cs336_data: {e}")
    print("Make sure the cs336_data module is correctly installed or in your Python path.")
    print("You might need to run: pip install -e .")
    sys.exit(1)


def extract_text_from_url(url: str) -> str:
    """
    Download content from a URL and extract plain text.
    
    Args:
        url: URL to extract text from
        
    Returns:
        Extracted text or empty string if failed
    """
    try:
        # Set a reasonable timeout
        with urllib.request.urlopen(url, timeout=5) as response:
            html_bytes = response.read()
            text = extract_text_from_html_bytes(html_bytes)
            return text if text else ""
    except Exception as e:
        return ""


def extract_common_crawl_samples(warc_file: str, num_samples: int = 1000) -> List[str]:
    """
    Extract random text samples from a Common Crawl WARC file.
    
    Args:
        warc_file: Path to WARC file
        num_samples: Number of samples to extract
        
    Returns:
        List of extracted text samples
    """
    samples = []
    try:
        with open(warc_file, 'rb') as f:
            for record in ArchiveIterator(f):
                # Only process response records with HTML content
                if record.rec_type == 'response' and record.http_headers and \
                   record.http_headers.get_header('Content-Type', '').startswith('text/html'):
                    
                    content = record.content_stream().read()
                    
                    try:
                        # Extract text from HTML content
                        text = extract_text_from_html_bytes(content)
                        
                        # Skip empty or very short texts
                        if text and len(text.strip()) > 100:
                            # Check language - only keep English
                            lang, confidence = identify_language(text)
                            if lang == "en" and confidence > 0.8:
                                samples.append(text)
                        
                        # Break early if we have enough samples
                        if len(samples) >= num_samples:
                            break
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error processing WARC file: {e}")
    
    return samples


def extract_wiki_reference_samples(wiki_urls_file: str, num_samples: int = 1000) -> List[str]:
    """
    Extract text samples from Wikipedia reference URLs.
    
    Args:
        wiki_urls_file: File containing Wikipedia reference URLs
        num_samples: Number of samples to extract
        
    Returns:
        List of extracted text samples
    """
    samples = []
    urls = []
    
    # First, extract and shuffle URLs
    try:
        if wiki_urls_file.endswith('.gz'):
            with gzip.open(wiki_urls_file, 'rt', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            with open(wiki_urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
        
        # Shuffle to get a random sample
        random.shuffle(urls)
        
    except Exception as e:
        print(f"Error reading Wiki URLs file: {e}")
        return samples
    
    # Try to download and extract text from URLs
    count = 0
    for url in urls:
        if not url.startswith(('http://', 'https://')):
            continue
            
        text = extract_text_from_url(url)
        if text and len(text.strip()) > 100:
            # Check language - only keep English
            lang, confidence = identify_language(text)
            if lang == "en" and confidence > 0.8:
                # Apply Gopher quality filters to ensure we have high quality examples
                if gopher_quality_filter(text):
                    samples.append(text)
                    count += 1
                    
                    # Print progress
                    if count % 10 == 0:
                        print(f"Processed {count} wiki samples")
                        
                    # Stop when we have enough samples
                    if count >= num_samples:
                        break
    
    return samples


def clean_text_for_fasttext(text: str) -> str:
    """
    Clean and normalize text for fastText training.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace quotes that could cause parsing issues
    text = text.replace('"', ' ')
    
    return text


def create_training_data(wiki_samples: List[str], cc_samples: List[str], output_file: str):
    """
    Create a fastText training file with labeled examples.
    
    Args:
        wiki_samples: List of high-quality wiki samples
        cc_samples: List of low-quality CC samples
        output_file: Path to write training data
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write high-quality examples (wiki)
        for text in wiki_samples:
            cleaned_text = clean_text_for_fasttext(text)
            # Take first 1000 chars to avoid huge training examples
            if cleaned_text:
                f.write(f"__label__wiki {cleaned_text[:1000]}\n")
        
        # Write low-quality examples (cc)
        for text in cc_samples:
            cleaned_text = clean_text_for_fasttext(text)
            if cleaned_text:
                f.write(f"__label__cc {cleaned_text[:1000]}\n")


def train_fasttext_model(train_file: str, output_model: str):
    """
    Train a fastText classifier on labeled data.
    
    Args:
        train_file: Path to training data file
        output_model: Path to save trained model
    """
    # Create a balanced train/test split
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = f.readlines()
    
    random.shuffle(train_data)
    split_point = int(len(train_data) * 0.8)
    
    # Write train/test files
    temp_dir = tempfile.mkdtemp()
    train_temp = os.path.join(temp_dir, "train.txt")
    test_temp = os.path.join(temp_dir, "test.txt")
    
    with open(train_temp, 'w', encoding='utf-8') as f:
        f.writelines(train_data[:split_point])
    
    with open(test_temp, 'w', encoding='utf-8') as f:
        f.writelines(train_data[split_point:])
    
    # Train the model
    model = fasttext.train_supervised(
        train_temp,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=100,
        loss='softmax'
    )
    
    # Evaluate on test data
    result = model.test(test_temp)
    print(f"Test set evaluation:")
    print(f"Samples: {result[0]}")
    print(f"Precision: {result[1]}")
    print(f"Recall: {result[2]}")
    
    # Save the model
    model.save_model(output_model)
    
    # Cleanup temp files
    shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description='Train a fastText quality classifier')
    parser.add_argument('--cc-warc', help='Path to Common Crawl WARC file for low-quality examples')
    parser.add_argument('--wiki-urls', help='Path to Wikipedia reference URLs file for high-quality examples')
    parser.add_argument('--cc-samples', type=int, default=1000, help='Number of CC samples to extract')
    parser.add_argument('--wiki-samples', type=int, default=1000, help='Number of Wiki samples to extract')
    parser.add_argument('--output-model', default='quality_classifier.bin', help='Output model file')
    parser.add_argument('--use-test-data', action='store_true', help='Use test fixtures instead of real data')
    args = parser.parse_args()
    
    # If using test data
    if args.use_test_data:
        # Find fixtures path
        project_root = Path(__file__).parent.parent
        fixtures_path = project_root / "tests" / "fixtures"
        
        # Load test fixtures
        try:
            with open(fixtures_path / "high_quality_wiki_reference.txt", 'r') as f:
                wiki_text = f.read()
            
            with open(fixtures_path / "low_quality_cc.txt", 'r') as f:
                cc_text = f.read()
                
            wiki_samples = [wiki_text]
            cc_samples = [cc_text]
            
            # Add variations by taking sections
            wiki_paragraphs = wiki_text.split('\n\n')
            cc_paragraphs = cc_text.split('\n\n')
            
            for i in range(0, len(wiki_paragraphs) - 3, 3):
                sample = '\n\n'.join(wiki_paragraphs[i:i+3])
                if len(sample) > 100:
                    wiki_samples.append(sample)
            
            for i in range(0, len(cc_paragraphs) - 2, 2):
                sample = '\n\n'.join(cc_paragraphs[i:i+2])
                if len(sample) > 20:
                    cc_samples.append(sample)
            
            # Create the models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_model)))
            os.makedirs(models_dir, exist_ok=True)
            
            # Create training data file
            train_file = os.path.join(models_dir, "quality_train.txt")
            create_training_data(wiki_samples, cc_samples, train_file)
            
            # Train the model
            train_fasttext_model(train_file, args.output_model)
            
            # Clean up
            if os.path.exists(train_file):
                os.remove(train_file)
                
            print(f"Model trained successfully using test data and saved to {args.output_model}")
                
        except Exception as e:
            print(f"Error using test data: {e}")
            sys.exit(1)
    
    else:
        # Using real data
        if not args.cc_warc or not args.wiki_urls:
            print("Error: Both --cc-warc and --wiki-urls must be provided")
            sys.exit(1)
        
        # Extract samples
        print(f"Extracting {args.cc_samples} samples from Common Crawl...")
        cc_samples = extract_common_crawl_samples(args.cc_warc, args.cc_samples)
        print(f"Extracted {len(cc_samples)} Common Crawl samples")
        
        print(f"Extracting {args.wiki_samples} samples from Wikipedia references...")
        wiki_samples = extract_wiki_reference_samples(args.wiki_urls, args.wiki_samples)
        print(f"Extracted {len(wiki_samples)} Wikipedia samples")
        
        # Create models directory if needed
        models_dir = os.path.dirname(os.path.abspath(args.output_model))
        os.makedirs(models_dir, exist_ok=True)
        
        # Create training data file
        train_file = os.path.join(models_dir, "quality_train.txt")
        create_training_data(wiki_samples, cc_samples, train_file)
        
        # Train the model
        train_fasttext_model(train_file, args.output_model)
        
        # Clean up
        if os.path.exists(train_file):
            os.remove(train_file)
            
        print(f"Model trained successfully using {len(wiki_samples)} Wikipedia samples and {len(cc_samples)} Common Crawl samples")
        print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main() 