#!/usr/bin/env python3
"""
Script to run language identification on text extracted from WARC files.
This script selects 20 random samples, identifies their language, and displays
the results along with text snippets for manual verification.
"""

import os
import random
import argparse
import sys
from pathlib import Path

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
    from cs336_data.data import extract_text_from_html_bytes, identify_language
except ImportError as e:
    print(f"Error importing from cs336_data: {e}")
    print("Make sure the cs336_data module is correctly installed or in your Python path.")
    print("You might need to run: pip install -e .")
    sys.exit(1)


def extract_samples_from_warc(warc_path, num_samples=20, max_text_length=200):
    """
    Extract random text samples from a WARC file.
    
    Args:
        warc_path: Path to the WARC file
        num_samples: Number of random samples to extract
        max_text_length: Maximum length of text snippet to display
        
    Returns:
        List of tuples (text, url)
    """
    all_samples = []
    
    print(f"Reading WARC file: {warc_path}")
    
    try:
        with open(warc_path, 'rb') as f:
            for record in ArchiveIterator(f):
                # Only process response records with HTML content
                if record.rec_type == 'response' and record.http_headers and \
                   record.http_headers.get_header('Content-Type', '').startswith('text/html'):
                    
                    url = record.rec_headers.get_header('WARC-Target-URI')
                    content = record.content_stream().read()
                    
                    try:
                        # Extract text from HTML content
                        text = extract_text_from_html_bytes(content)
                        
                        # Skip empty or very short texts
                        if text and len(text.strip()) > 50:
                            all_samples.append((text, url))
                    except Exception as e:
                        print(f"Error processing {url}: {e}")
                
                # Break early if we have enough samples
                if len(all_samples) >= num_samples * 10:  # Collect more than needed to select random subset
                    break
    except FileNotFoundError:
        print(f"Error: The file {warc_path} does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening or processing WARC file: {e}")
        sys.exit(1)
    
    if not all_samples:
        print(f"No valid samples found in {warc_path}")
        print("Make sure the file is a valid WARC file with HTML content.")
        sys.exit(1)
    
    # Select random samples
    if len(all_samples) > num_samples:
        selected_samples = random.sample(all_samples, num_samples)
    else:
        print(f"Warning: Found only {len(all_samples)} valid samples instead of the requested {num_samples}.")
        selected_samples = all_samples
    
    return selected_samples


def process_samples(samples, max_text_length=200):
    """
    Process text samples by identifying their language and preparing display info.
    
    Args:
        samples: List of tuples (text, url)
        max_text_length: Maximum length of text snippet to display
        
    Returns:
        Dictionary with statistics and processed samples
    """
    results = []
    language_counts = {}
    confidence_values = []
    
    for i, (text, url) in enumerate(samples, 1):
        # Identify language
        lang_code, confidence = identify_language(text)
        
        # Update statistics
        language_counts[lang_code] = language_counts.get(lang_code, 0) + 1
        confidence_values.append(confidence)
        
        # Prepare text snippet (first few characters)
        snippet = text.strip().replace('\n', ' ')[:max_text_length]
        if len(text) > max_text_length:
            snippet += "..."
        
        # Store result
        results.append({
            'sample_num': i,
            'lang_code': lang_code,
            'confidence': confidence,
            'snippet': snippet,
            'url': url
        })
    
    # Calculate English percentage
    english_percent = language_counts.get('en', 0) / len(samples) * 100 if samples else 0
    
    # Calculate average confidence
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    
    return {
        'samples': results,
        'language_counts': language_counts,
        'english_percent': english_percent,
        'avg_confidence': avg_confidence
    }


def main():
    parser = argparse.ArgumentParser(description='Run language identification on WARC files')
    parser.add_argument('warc_file', help='Path to WARC file')
    parser.add_argument('--samples', type=int, default=20, help='Number of random samples to process')
    args = parser.parse_args()
    
    # Extract samples from WARC file
    samples = extract_samples_from_warc(args.warc_file, args.samples)
    
    if not samples:
        print(f"No valid samples found in {args.warc_file}")
        return
    
    # Process samples
    results = process_samples(samples)
    
    # Display results
    print("\n" + "="*80)
    print(f"LANGUAGE IDENTIFICATION RESULTS FOR {args.samples} RANDOM SAMPLES")
    print("="*80)
    
    for sample in results['samples']:
        print(f"\nSample {sample['sample_num']}")
        print(f"Language: {sample['lang_code']} (Confidence: {sample['confidence']:.2f})")
        print(f"URL: {sample['url']}")
        print("-"*40)
        print(f"Text Snippet: {sample['snippet']}")
        print("-"*80)
    
    # Display statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print("Language distribution:")
    for lang, count in results['language_counts'].items():
        print(f"  {lang}: {count} samples ({count / len(samples) * 100:.1f}%)")
    
    print(f"\nPercentage of English documents: {results['english_percent']:.1f}%")
    print(f"Average confidence score: {results['avg_confidence']:.2f}")
    
    print("\nBased on these samples, analyze the results to determine:")
    print("1. If the language identification is accurate")
    print("2. What confidence threshold would be suitable for filtering")
    

if __name__ == "__main__":
    main() 