#!/usr/bin/env python3
"""
Script to run PII masking on text extracted from WARC files.
This script finds examples where PII (emails, phone numbers, IPs) was masked
and displays the context to help identify false positives and false negatives.
"""

import os
import random
import argparse
import sys
import re
from pathlib import Path
from warcio.archiveiterator import ArchiveIterator
from cs336_data.data import (
    extract_text_from_html_bytes,
    mask_emails,
    mask_phone_numbers,
    mask_ips
)


def extract_samples_from_warc(warc_path, sample_count=100):
    """
    Extract random text samples from a WARC file.
    
    Args:
        warc_path: Path to the WARC file
        sample_count: Number of samples to extract (we'll extract more for filtering)
        
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
                        if text and len(text.strip()) > 100:
                            all_samples.append((text, url))
                    except Exception as e:
                        print(f"Error processing {url}: {e}")
                
                # Break early if we have enough samples
                if len(all_samples) >= sample_count * 5:  # Collect more than needed for filtering
                    break
    except Exception as e:
        print(f"Error processing WARC file: {e}")
        sys.exit(1)
    
    # Select random samples
    if all_samples:
        if len(all_samples) > sample_count:
            return random.sample(all_samples, sample_count)
        return all_samples
    else:
        print("No valid samples found in the WARC file.")
        sys.exit(1)


def extract_context(text, masked_text, mask_placeholder, context_chars=50):
    """
    Extract context around where masking occurred.
    
    Args:
        text: Original text
        masked_text: Text after masking
        mask_placeholder: The placeholder string used for masking
        context_chars: Number of characters of context to include on each side
        
    Returns:
        List of tuples (original_fragment, masked_fragment)
    """
    masks = []
    
    # Find all occurrences of the mask placeholder
    for match in re.finditer(re.escape(mask_placeholder), masked_text):
        start, end = match.span()
        
        # Determine the context bounds
        context_start = max(0, start - context_chars)
        context_end = min(len(masked_text), end + context_chars)
        
        # Get the masked fragment with context
        masked_fragment = masked_text[context_start:context_end]
        
        # Try to find what was masked in the original text
        original_context_before = masked_text[context_start:start]
        original_context_after = masked_text[end:context_end]
        
        # Find original text by matching the context before and after
        original_matches = []
        for i in range(len(text)):
            if text[i:].startswith(original_context_before):
                # Found the start of the context
                match_start = i
                match_after_start = match_start + len(original_context_before)
                
                # Find where the after context appears
                potential_end = text.find(original_context_after, match_after_start)
                if potential_end > 0:
                    # Original content between before and after context
                    original_pii = text[match_after_start:potential_end]
                    if len(original_pii) > 0 and len(original_pii) < 100:  # Sanity check
                        original_fragment = text[match_start:potential_end + len(original_context_after)]
                        original_matches.append((original_fragment, original_pii))
                        break
        
        if original_matches:
            masks.append((original_matches[0][0], masked_fragment, original_matches[0][1]))
        else:
            # Fallback if we can't find the exact match
            masks.append((None, masked_fragment, "UNKNOWN"))
    
    return masks


def mask_and_analyze_pii(samples):
    """
    Mask PII in the samples and analyze the results.
    
    Args:
        samples: List of tuples (text, url)
        
    Returns:
        Dictionary with masking results
    """
    results = {
        'emails': [],
        'phones': [],
        'ips': []
    }
    
    for text, url in samples:
        # Mask emails
        masked_email_text, num_emails = mask_emails(text)
        if num_emails > 0:
            email_contexts = extract_context(text, masked_email_text, "|||EMAIL_ADDRESS|||")
            for orig, masked, pii in email_contexts:
                results['emails'].append({
                    'url': url,
                    'original': orig,
                    'masked': masked,
                    'pii': pii
                })
        
        # Mask phone numbers
        masked_phone_text, num_phones = mask_phone_numbers(text)
        if num_phones > 0:
            phone_contexts = extract_context(text, masked_phone_text, "|||PHONE_NUMBER|||")
            for orig, masked, pii in phone_contexts:
                results['phones'].append({
                    'url': url,
                    'original': orig,
                    'masked': masked,
                    'pii': pii
                })
        
        # Mask IP addresses
        masked_ip_text, num_ips = mask_ips(text)
        if num_ips > 0:
            ip_contexts = extract_context(text, masked_ip_text, "|||IP_ADDRESS|||")
            for orig, masked, pii in ip_contexts:
                results['ips'].append({
                    'url': url,
                    'original': orig,
                    'masked': masked,
                    'pii': pii
                })
    
    return results


def display_results(results, max_examples=20):
    """
    Display the PII masking results in a readable format.
    
    Args:
        results: Dictionary with masking results
        max_examples: Maximum number of examples to display per PII type
    """
    pii_types = {
        'emails': 'Email Addresses',
        'phones': 'Phone Numbers',
        'ips': 'IP Addresses'
    }
    
    for pii_type, pii_name in pii_types.items():
        examples = results[pii_type]
        
        print("\n" + "="*80)
        print(f"PII MASKING RESULTS: {pii_name} ({len(examples)} found)")
        print("="*80)
        
        if not examples:
            print(f"No {pii_name.lower()} were found in the samples.")
            continue
        
        # Select a random subset if we have too many examples
        if len(examples) > max_examples:
            display_examples = random.sample(examples, max_examples)
        else:
            display_examples = examples
        
        for i, example in enumerate(display_examples, 1):
            print(f"\nExample {i}/{len(display_examples)}")
            print(f"URL: {example['url']}")
            print(f"Masked PII: {example['pii']}")
            print("-"*40)
            
            if example['original']:
                print("Original Context:")
                print(example['original'])
                print("\nMasked Context:")
                print(example['masked'])
            else:
                print("Context:")
                print(example['masked'])
            
            print("-"*80)
        
        print(f"\nTotal {pii_name} found: {len(examples)}")
        
    print("\n" + "="*80)
    print("ANALYSIS NOTES")
    print("="*80)
    print("Examine the examples above to identify:")
    print("1. False positives: Items that were masked but are not actually PII")
    print("2. False negatives: PII that was not masked (look at the context)")
    print("\nSome common patterns:")
    print("- Email false positives: Text containing '@' that isn't an email")
    print("- Phone false positives: Numbers that look like phone numbers but aren't")
    print("- IP false positives: Number sequences that match IP format")
    print("- False negatives: PII in unusual formats not covered by our regexes")


def main():
    parser = argparse.ArgumentParser(description='Analyze PII masking on WARC files')
    parser.add_argument('warc_file', help='Path to WARC file')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to process')
    parser.add_argument('--examples', type=int, default=20, help='Number of examples to display per PII type')
    args = parser.parse_args()
    
    # Extract samples from WARC file
    samples = extract_samples_from_warc(args.warc_file, args.samples)
    
    # Mask and analyze PII
    results = mask_and_analyze_pii(samples)
    
    # Display results
    display_results(results, args.examples)


if __name__ == "__main__":
    main() 