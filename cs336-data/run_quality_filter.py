#!/usr/bin/env python3
"""
Script to run Gopher quality filters on text extracted from WARC files.
This script selects 20 random samples, applies the quality filters, and displays
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
    from cs336_data.data import extract_text_from_html_bytes, gopher_quality_filter
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
                        
                        # Skip very short texts
                        if text and len(text.strip()) > 10:
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
    Process text samples by applying Gopher quality filters and preparing display info.
    
    Args:
        samples: List of tuples (text, url)
        max_text_length: Maximum length of text snippet to display
        
    Returns:
        Dictionary with statistics and processed samples
    """
    results = []
    quality_counts = {"high_quality": 0, "low_quality": 0}
    
    # Count quality filter reasons
    filter_reasons = {
        "word_count": 0,
        "mean_word_length": 0,
        "ellipsis_lines": 0,
        "non_alpha_words": 0
    }
    
    for i, (text, url) in enumerate(samples, 1):
        # Apply Gopher quality filter
        is_quality = gopher_quality_filter(text)
        
        # Track specific filter failures for detailed analysis
        if not is_quality:
            # Get word count
            words = [word for word in text.split() if word]
            word_count = len(words)
            
            if word_count < 50 or word_count > 100000:
                filter_reasons["word_count"] += 1
            
            # Check mean word length
            alpha_words = [word for word in words if any(c.isalpha() for c in word)]
            if alpha_words:
                mean_word_length = sum(len(word) for word in alpha_words) / len(alpha_words)
                if mean_word_length < 3 or mean_word_length > 10:
                    filter_reasons["mean_word_length"] += 1
            
            # Check lines ending with ellipsis
            lines = text.strip().split('\n')
            if lines:
                ellipsis_percent = sum(1 for line in lines if line.strip().endswith('...')) / len(lines)
                if ellipsis_percent > 0.3:
                    filter_reasons["ellipsis_lines"] += 1
            
            # Check words with alphabetic characters
            if words:
                alpha_word_percent = sum(1 for word in words if any(c.isalpha() for c in word)) / len(words)
                if alpha_word_percent < 0.8:
                    filter_reasons["non_alpha_words"] += 1
        
        # Update quality counts
        if is_quality:
            quality_counts["high_quality"] += 1
        else:
            quality_counts["low_quality"] += 1
        
        # Prepare text snippet (first few characters)
        snippet = text.strip().replace('\n', ' ')[:max_text_length]
        if len(text) > max_text_length:
            snippet += "..."
        
        # Get text stats for display
        lines = text.strip().split('\n')
        words = [word for word in text.split() if word]
        alpha_words = [word for word in words if any(c.isalpha() for c in word)]
        
        stats = {
            "word_count": len(words),
            "line_count": len(lines),
            "avg_word_length": sum(len(word) for word in alpha_words) / len(alpha_words) if alpha_words else 0,
            "ellipsis_lines_pct": sum(1 for line in lines if line.strip().endswith('...')) / len(lines) if lines else 0,
            "alpha_words_pct": sum(1 for word in words if any(c.isalpha() for c in word)) / len(words) if words else 0
        }
        
        # Store result
        results.append({
            'sample_num': i,
            'quality': "High Quality" if is_quality else "Low Quality",
            'snippet': snippet,
            'url': url,
            'stats': stats
        })
    
    # Calculate high quality percentage
    high_quality_percent = quality_counts["high_quality"] / len(samples) * 100 if samples else 0
    
    return {
        'samples': results,
        'quality_counts': quality_counts,
        'high_quality_percent': high_quality_percent,
        'filter_reasons': filter_reasons
    }


def main():
    parser = argparse.ArgumentParser(description='Run Gopher quality filters on WARC files')
    parser.add_argument('warc_file', help='Path to WARC file')
    parser.add_argument('--samples', type=int, default=20, help='Number of random samples to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed text stats')
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
    print(f"QUALITY FILTER RESULTS FOR {args.samples} RANDOM SAMPLES")
    print("="*80)
    
    for sample in results['samples']:
        print(f"\nSample {sample['sample_num']}")
        print(f"Quality: {sample['quality']}")
        print(f"URL: {sample['url']}")
        
        # Print stats if verbose mode
        if args.verbose:
            print("Text Statistics:")
            print(f"  Word Count: {sample['stats']['word_count']}")
            print(f"  Line Count: {sample['stats']['line_count']}")
            print(f"  Avg Word Length: {sample['stats']['avg_word_length']:.2f}")
            print(f"  % Lines Ending with Ellipsis: {sample['stats']['ellipsis_lines_pct']*100:.1f}%")
            print(f"  % Words with Alphabetic Chars: {sample['stats']['alpha_words_pct']*100:.1f}%")
        
        print("-"*40)
        print(f"Text Snippet: {sample['snippet']}")
        print("-"*80)
    
    # Display statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print("Quality distribution:")
    print(f"  High Quality: {results['quality_counts']['high_quality']} samples ({results['high_quality_percent']:.1f}%)")
    print(f"  Low Quality: {results['quality_counts']['low_quality']} samples ({100-results['high_quality_percent']:.1f}%)")
    
    # Display filter reason statistics
    if results['quality_counts']['low_quality'] > 0:
        print("\nReasons for low quality:")
        for reason, count in results['filter_reasons'].items():
            if count > 0:
                percent = count / results['quality_counts']['low_quality'] * 100
                print(f"  {reason.replace('_', ' ').title()}: {count} samples ({percent:.1f}% of low quality)")
    
    print("\nBased on these samples, analyze the results to determine:")
    print("1. If the quality filters align with your judgment of text quality")
    print("2. Which specific filters are most effectively identifying low-quality content")
    print("3. Are there cases where the filters misclassify high-quality content as low-quality (or vice versa)?")
    

if __name__ == "__main__":
    main() 