#!/usr/bin/env python3
"""
Script to analyze harmful content in a specific WARC file.
"""

import sys
import random
import gzip
from warcio.archiveiterator import ArchiveIterator
from cs336_data.data import (
    extract_text_from_html_bytes,
    classify_nsfw,
    classify_toxic_speech
)

# Path to the WARC file
WARC_FILE = "/Users/lenox/Desktop/ECE491B/assignment2/s2025-assignment2-data/data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz"
NUM_SAMPLES = 20
MIN_TEXT_LENGTH = 100

def main():
    """
    Main function to analyze harmful content in the WARC file.
    """
    print(f"Analyzing WARC file: {WARC_FILE}")
    
    # List to store all extracted texts
    all_texts = []
    
    # Open the gzipped WARC file
    with gzip.open(WARC_FILE, 'rb') as warc_file:
        # Process each record in the WARC file
        for record in ArchiveIterator(warc_file):
            # Only process response records with HTML content
            if (record.rec_type == 'response' and 
                record.http_headers and 
                record.http_headers.get_header('Content-Type') and 
                'text/html' in record.http_headers.get_header('Content-Type').lower()):
                
                # Extract text from HTML content
                try:
                    html_content = record.content_stream().read()
                    text = extract_text_from_html_bytes(html_content)
                    
                    # Skip texts that are too short
                    if len(text) < MIN_TEXT_LENGTH:
                        continue
                    
                    # Store the extracted text
                    all_texts.append(text)
                except Exception as e:
                    print(f"Error extracting text: {e}")
                    continue
    
    print(f"Total documents with text: {len(all_texts)}")
    
    # If we have enough texts, select random samples
    if len(all_texts) > 0:
        # Choose random samples (or all if fewer than requested)
        sample_size = min(NUM_SAMPLES, len(all_texts))
        selected_texts = random.sample(all_texts, sample_size)
        
        # Track statistics
        nsfw_count = 0
        toxic_count = 0
        harmful_count = 0
        
        # Confidence threshold counters
        nsfw_thresholds = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
        toxic_thresholds = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
        
        # Analyze each sample
        print("\n" + "="*80)
        print(" HARMFUL CONTENT ANALYSIS RESULTS ".center(80, "="))
        print("="*80 + "\n")
        
        for i, text in enumerate(selected_texts, 1):
            # Truncate text for display
            display_text = text[:500] + "..." if len(text) > 500 else text
            
            # Classify the text
            nsfw_result = classify_nsfw(text)
            toxic_result = classify_toxic_speech(text)
            
            nsfw_label, nsfw_confidence = nsfw_result
            toxic_label, toxic_confidence = toxic_result
            
            # Update statistics
            if nsfw_label == "nsfw":
                nsfw_count += 1
                # Update threshold counters
                for threshold in nsfw_thresholds:
                    if nsfw_confidence >= threshold:
                        nsfw_thresholds[threshold] += 1
            
            if toxic_label == "toxic":
                toxic_count += 1
                # Update threshold counters
                for threshold in toxic_thresholds:
                    if toxic_confidence >= threshold:
                        toxic_thresholds[threshold] += 1
            
            if nsfw_label == "nsfw" or toxic_label == "toxic":
                harmful_count += 1
            
            # Print the sample
            print(f"SAMPLE {i}:")
            print(f"Text: {display_text}")
            print(f"NSFW: {nsfw_label} (confidence: {nsfw_confidence:.4f})")
            print(f"Toxic: {toxic_label} (confidence: {toxic_confidence:.4f})")
            print("-"*80)
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print(f"Total samples analyzed: {sample_size}")
        print(f"NSFW content: {nsfw_count} ({nsfw_count/sample_size*100:.1f}%)")
        print(f"Toxic content: {toxic_count} ({toxic_count/sample_size*100:.1f}%)")
        print(f"Harmful content (NSFW or toxic): {harmful_count} ({harmful_count/sample_size*100:.1f}%)")
        
        # Print threshold analysis
        print("\nNSFW THRESHOLD ANALYSIS:")
        for threshold in sorted(nsfw_thresholds.keys()):
            count = nsfw_thresholds[threshold]
            print(f"  >= {threshold:.1f}: {count} samples ({count/sample_size*100:.1f}%)")
        
        print("\nTOXIC THRESHOLD ANALYSIS:")
        for threshold in sorted(toxic_thresholds.keys()):
            count = toxic_thresholds[threshold]
            print(f"  >= {threshold:.1f}: {count} samples ({count/sample_size*100:.1f}%)")
        
        print("\nRECOMMENDATIONS FOR MANUAL VERIFICATION:")
        print("1. Compare the classifier predictions to your own judgment.")
        print("2. Note any classifier errors (false positives or false negatives).")
        print("3. Based on your observations, determine suitable confidence thresholds.")
        print("4. Consider the trade-off between filtering harmful content and preserving non-harmful content.")
    else:
        print("No text samples were found in the WARC file.")

if __name__ == "__main__":
    main() 