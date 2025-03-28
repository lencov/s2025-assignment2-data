#!/usr/bin/env python3
"""
Script to analyze WARC files for harmful content.

This script extracts text from WARC files, analyzes it for harmful content
(NSFW and toxic speech), and outputs the results for manual inspection.
"""

import os
import sys
import random
import gzip
from pathlib import Path
from datetime import datetime
from collections import Counter

from warcio.archiveiterator import ArchiveIterator
from cs336_data.data import (
    extract_text_from_html_bytes,
    classify_nsfw,
    classify_toxic_speech
)


def analyze_warc_file(warc_path, num_samples=20, min_text_length=100):
    """
    Analyze a WARC file for harmful content.
    
    Args:
        warc_path: Path to the WARC file
        num_samples: Number of random samples to analyze
        min_text_length: Minimum text length to consider for analysis
    
    Returns:
        List of tuples (text, nsfw_result, toxic_result)
    """
    print(f"Analyzing WARC file: {warc_path}")
    
    # List to store text samples and their analysis results
    samples = []
    all_texts = []
    
    # Process the WARC file
    open_func = gzip.open if str(warc_path).endswith('.gz') else open
    
    with open_func(warc_path, 'rb') as warc_file:
        for record in ArchiveIterator(warc_file):
            # Only process response records with HTML content
            if (record.rec_type == 'response' and 
                record.http_headers and 
                record.http_headers.get_header('Content-Type') and 
                'text/html' in record.http_headers.get_header('Content-Type').lower()):
                
                # Extract text from HTML content
                html_content = record.content_stream().read()
                try:
                    text = extract_text_from_html_bytes(html_content)
                    
                    # Skip texts that are too short
                    if len(text) < min_text_length:
                        continue
                    
                    # Store the extracted text
                    all_texts.append(text)
                except Exception as e:
                    print(f"Error extracting text: {e}")
                    continue
    
    # If we have enough texts, select random samples
    if len(all_texts) > 0:
        # Choose random samples (or all if fewer than requested)
        sample_size = min(num_samples, len(all_texts))
        selected_texts = random.sample(all_texts, sample_size)
        
        # Analyze each selected text
        for text in selected_texts:
            # Truncate text for display
            display_text = text[:500] + "..." if len(text) > 500 else text
            
            # Classify the text
            nsfw_result = classify_nsfw(text)
            toxic_result = classify_toxic_speech(text)
            
            # Add to samples
            samples.append((display_text, nsfw_result, toxic_result))
    
    return samples, len(all_texts)


def print_analysis_results(samples):
    """
    Print the analysis results in a readable format.
    
    Args:
        samples: List of tuples (text, nsfw_result, toxic_result)
    """
    print("\n" + "="*80)
    print(" HARMFUL CONTENT ANALYSIS RESULTS ".center(80, "="))
    print("="*80 + "\n")
    
    # Track statistics
    nsfw_count = 0
    toxic_count = 0
    nsfw_confidence_sum = 0
    toxic_confidence_sum = 0
    
    # Confidence threshold counters
    nsfw_thresholds = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    toxic_thresholds = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    
    # Print each sample with its classification
    for i, (text, nsfw_result, toxic_result) in enumerate(samples, 1):
        nsfw_label, nsfw_confidence = nsfw_result
        toxic_label, toxic_confidence = toxic_result
        
        # Update statistics
        if nsfw_label == "nsfw":
            nsfw_count += 1
            nsfw_confidence_sum += nsfw_confidence
            # Update threshold counters
            for threshold in nsfw_thresholds:
                if nsfw_confidence >= threshold:
                    nsfw_thresholds[threshold] += 1
        
        if toxic_label == "toxic":
            toxic_count += 1
            toxic_confidence_sum += toxic_confidence
            # Update threshold counters
            for threshold in toxic_thresholds:
                if toxic_confidence >= threshold:
                    toxic_thresholds[threshold] += 1
        
        # Print the sample
        print(f"SAMPLE {i}:")
        print(f"Text: {text}")
        print(f"NSFW: {nsfw_label} (confidence: {nsfw_confidence:.4f})")
        print(f"Toxic: {toxic_label} (confidence: {toxic_confidence:.4f})")
        print("-"*80)
    
    # Print summary statistics
    total_samples = len(samples)
    print("\nSUMMARY STATISTICS:")
    print(f"Total samples analyzed: {total_samples}")
    print(f"NSFW content: {nsfw_count} ({nsfw_count/total_samples*100:.1f}%)")
    print(f"Toxic content: {toxic_count} ({toxic_count/total_samples*100:.1f}%)")
    
    # Print threshold analysis
    print("\nNSFW THRESHOLD ANALYSIS:")
    for threshold, count in sorted(nsfw_thresholds.items()):
        print(f"  >= {threshold:.1f}: {count} samples ({count/total_samples*100:.1f}%)")
    
    print("\nTOXIC THRESHOLD ANALYSIS:")
    for threshold, count in sorted(toxic_thresholds.items()):
        print(f"  >= {threshold:.1f}: {count} samples ({count/total_samples*100:.1f}%)")
    
    print("\nNOTE: Please manually verify these classifications and note any errors.")
    print("Based on your observations, determine suitable confidence thresholds.")


def main():
    """
    Main function to run the harmful content analysis.
    """
    # Hard-coded path to the specific WARC file
    warc_path = Path("/Users/lenox/Desktop/ECE491B/assignment2/s2025-assignment2-data/data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz")
    
    # Fixed parameters
    num_samples = 20
    min_text_length = 100
    
    all_samples = []
    total_documents = 0
    
    # Create samples file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"harmful_content_analysis_{timestamp}.txt"
    
    # Check if path exists
    if not warc_path.exists():
        print(f"Error: Path {warc_path} does not exist.")
        sys.exit(1)
    
    print(f"Analyzing WARC file: {warc_path}")
    
    # Process the WARC file
    samples, doc_count = analyze_warc_file(warc_path, num_samples, min_text_length)
    all_samples.extend(samples)
    total_documents += doc_count
    
    # Print results
    if all_samples:
        # Save to file and print to console
        with open(output_file, 'w', encoding='utf-8') as f:
            # Redirect stdout to file
            original_stdout = sys.stdout
            sys.stdout = f
            
            print_analysis_results(all_samples)
            
            # Restore stdout
            sys.stdout = original_stdout
        
        print(f"\nResults saved to {output_file}")
        print_analysis_results(all_samples)
        
        print(f"\nTotal documents processed: {total_documents}")
        harmful_count = sum(1 for _, (nsfw_label, _), (toxic_label, _) in all_samples if nsfw_label == "nsfw" or toxic_label == "toxic")
        print(f"Documents with harmful content: {harmful_count} ({harmful_count/len(all_samples)*100:.1f}%)")
    else:
        print("No samples were found or analyzed.")


if __name__ == "__main__":
    main() 