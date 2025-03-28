#!/usr/bin/env python3
"""
Script to run analysis on existing WARC files.
This script uses a specified WARC file or searches for WARC files in a directory
and runs various analyses on them.
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path

def find_warc_files(directory="."):
    """Find WARC files in the specified directory."""
    warc_patterns = ["*.warc", "*.warc.gz"]
    warc_files = []
    
    for pattern in warc_patterns:
        warc_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return warc_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run analysis on WARC files')
    parser.add_argument('--file', help='Path to specific WARC file to analyze')
    parser.add_argument('--dir', default='.', help='Directory to search for WARC files')
    parser.add_argument('--samples', type=int, default=20, 
                        help='Number of random samples to process')
    parser.add_argument('--analysis', choices=['language', 'pii'], default='language',
                        help='Type of analysis to run (language or pii)')
    args = parser.parse_args()
    
    # Make sure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    warc_files = []
    
    # Use specific file if provided
    if args.file and os.path.exists(args.file):
        warc_files = [args.file]
    # Otherwise search for WARC files in the specified directory
    else:
        warc_files = find_warc_files(args.dir)
    
    if not warc_files:
        print("No WARC files found. Please specify a valid file with --file or a directory with --dir")
        sys.exit(1)
    
    # Print found WARC files
    print(f"Found {len(warc_files)} WARC file(s):")
    for i, file in enumerate(warc_files, 1):
        print(f"{i}. {file}")
    
    # If multiple files are found, let the user choose which one to analyze
    chosen_file = warc_files[0]
    if len(warc_files) > 1:
        while True:
            try:
                choice = input("\nEnter the number of the file to analyze (or press Enter to use the first one): ")
                if not choice:
                    break
                choice = int(choice)
                if 1 <= choice <= len(warc_files):
                    chosen_file = warc_files[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(warc_files)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Run the appropriate analysis script
    if args.analysis == 'language':
        print(f"\nRunning language identification analysis on {chosen_file}...")
        cmd = [sys.executable, "run_language_identification.py", chosen_file]
        if args.samples != 20:
            cmd.extend(["--samples", str(args.samples)])
        subprocess.run(cmd)
    elif args.analysis == 'pii':
        print(f"\nRunning PII masking analysis on {chosen_file}...")
        cmd = [sys.executable, "run_pii_masking.py", chosen_file]
        if args.samples != 20:
            cmd.extend(["--samples", str(args.samples)])
        subprocess.run(cmd)

if __name__ == "__main__":
    main() 