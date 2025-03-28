#!/usr/bin/env python3
"""
Module for document deduplication methods.
"""

import os
import hashlib
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set
import re
import random
import unicodedata
import mmh3  # MurmurHash3 for fast hashing


def hash_line(line: str) -> str:
    """
    Create a hash for a line of text.
    
    Args:
        line: The line to hash
        
    Returns:
        A string representation of the hash
    """
    return hashlib.md5(line.encode('utf-8')).hexdigest()


def count_line_frequency(file_paths: List[os.PathLike]) -> Dict[str, int]:
    """
    Count the frequency of each line (using hashes) across all input files.
    
    Args:
        file_paths: List of paths to input files
        
    Returns:
        Dictionary mapping line hashes to their frequency
    """
    line_counter = Counter()
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Use hash of the line as the key
                    line_hash = hash_line(line)
                    line_counter[line_hash] += 1
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not process {file_path}: {e}")
    
    return line_counter


def exact_line_deduplication(input_files: List[os.PathLike], output_directory: os.PathLike):
    """
    Perform exact line deduplication on a list of input files.
    
    The function:
    1. Counts the frequency of each line across all input files
    2. Rewrites each file to the output directory, keeping only lines that are unique in the corpus
    
    Args:
        input_files: List of paths to input files
        output_directory: Path to output directory where deduplicated files will be written
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count line frequencies across all files
    line_frequencies = count_line_frequency(input_files)
    
    # Keep track of which lines to keep (those with frequency 1)
    unique_line_hashes = {line_hash for line_hash, freq in line_frequencies.items() if freq == 1}
    
    # Process each file and write deduplicated versions
    for input_file in input_files:
        input_path = Path(input_file)
        output_path = output_dir / input_path.name
        
        try:
            # Read input file and filter out non-unique lines
            with open(input_path, 'r', encoding='utf-8') as in_file, \
                 open(output_path, 'w', encoding='utf-8') as out_file:
                
                for line in in_file:
                    line_hash = hash_line(line)
                    # Only keep lines that appear exactly once in the corpus
                    if line_hash in unique_line_hashes:
                        out_file.write(line)
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Error processing {input_file}: {e}") 


def normalize_text(text: str) -> str:
    """
    Normalize text for fuzzy matching.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Apply NFD unicode normalization
    text = unicodedata.normalize('NFD', text)
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace (replace multiple whitespace with single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove accents
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    return text


def create_ngrams(text: str, n: int) -> Set[str]:
    """
    Create word n-grams from text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        Set of n-gram strings
    """
    # Split text into words
    words = text.split()
    
    # Create n-grams
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.add(ngram)
    
    return ngrams


def compute_minhash_signature(ngrams: Set[str], num_hashes: int, seed: int = 42) -> List[int]:
    """
    Compute MinHash signature for a set of n-grams.
    
    Args:
        ngrams: Set of n-grams
        num_hashes: Number of hash functions to use
        seed: Seed for random hash functions
        
    Returns:
        List of minhash values (signature)
    """
    if not ngrams:
        return [0] * num_hashes
    
    # Initialize signature with maximum possible hash value
    signature = [float('inf')] * num_hashes
    
    # For each n-gram, compute hashes and keep the minimum for each hash function
    for ngram in ngrams:
        for i in range(num_hashes):
            # Use different seeds for each hash function
            hash_value = mmh3.hash(ngram, seed + i)
            signature[i] = min(signature[i], hash_value)
    
    return signature


def apply_lsh(signatures: Dict[str, List[int]], num_bands: int) -> Dict[int, List[str]]:
    """
    Apply Locality-Sensitive Hashing (LSH) to group similar documents.
    
    Args:
        signatures: Dictionary mapping document paths to their minhash signatures
        num_bands: Number of bands to divide signatures into
        
    Returns:
        Dictionary mapping band hashes to lists of document paths
    """
    # Dictionary to store document paths by band hash
    buckets = {}
    
    # Calculate rows per band
    if not signatures:
        return buckets
        
    num_hashes = len(next(iter(signatures.values())))
    rows_per_band = num_hashes // num_bands
    
    # For each document
    for doc_path, signature in signatures.items():
        # For each band
        for band_idx in range(num_bands):
            # Extract the portion of the signature for this band
            start_idx = band_idx * rows_per_band
            end_idx = start_idx + rows_per_band
            band = tuple(signature[start_idx:end_idx])
            
            # Create a hash for this band
            band_hash = hash((band_idx, band))
            
            # Add document to the appropriate bucket
            if band_hash not in buckets:
                buckets[band_hash] = []
            buckets[band_hash].append(doc_path)
    
    return buckets


def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity (intersection size / union size)
    """
    if not set1 and not set2:
        return 1.0  # Both empty sets are identical
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union


def find_duplicate_clusters(
    candidates: Dict[int, List[str]], 
    doc_ngrams: Dict[str, Set[str]], 
    jaccard_threshold: float
) -> List[List[str]]:
    """
    Find clusters of duplicate documents based on Jaccard similarity.
    
    Args:
        candidates: Dictionary mapping bucket hashes to lists of document paths
        doc_ngrams: Dictionary mapping document paths to their n-grams
        jaccard_threshold: Minimum Jaccard similarity to consider documents as duplicates
        
    Returns:
        List of document clusters (each cluster is a list of document paths)
    """
    # Dictionary to track duplicate pairs
    duplicates = {}
    
    # For each bucket, check all pairs for similarity
    for bucket, doc_paths in candidates.items():
        if len(doc_paths) < 2:
            continue
            
        # Check all pairs in this bucket
        for i in range(len(doc_paths)):
            for j in range(i + 1, len(doc_paths)):
                doc1 = doc_paths[i]
                doc2 = doc_paths[j]
                
                # Skip if we've already identified these as duplicates
                if doc1 in duplicates and doc2 in duplicates[doc1]:
                    continue
                
                # Calculate actual Jaccard similarity
                similarity = compute_jaccard_similarity(doc_ngrams[doc1], doc_ngrams[doc2])
                
                # If similarity exceeds threshold, mark as duplicates
                if similarity >= jaccard_threshold:
                    if doc1 not in duplicates:
                        duplicates[doc1] = set()
                    if doc2 not in duplicates:
                        duplicates[doc2] = set()
                    
                    duplicates[doc1].add(doc2)
                    duplicates[doc2].add(doc1)
    
    # Use a set to track which documents we've already processed
    processed = set()
    clusters = []
    
    # For each document, find its cluster
    for doc in doc_ngrams.keys():
        if doc in processed:
            continue
            
        # Start a new cluster with this document
        cluster = [doc]
        processed.add(doc)
        
        # Find all duplicates using BFS
        queue = list(duplicates.get(doc, []))
        while queue:
            next_doc = queue.pop(0)
            if next_doc in processed:
                continue
                
            cluster.append(next_doc)
            processed.add(next_doc)
            
            # Add all duplicates of next_doc to queue
            queue.extend([d for d in duplicates.get(next_doc, []) if d not in processed])
        
        # Only add clusters with more than one document
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters


def minhash_deduplication(
    input_files: List[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike
):
    """
    Perform fuzzy document deduplication using MinHash and LSH.
    
    Args:
        input_files: List of paths to input files
        num_hashes: Number of hash functions to use for MinHash signatures
        num_bands: Number of bands to use for LSH
        ngrams: Size of n-grams (in words)
        jaccard_threshold: Minimum Jaccard similarity to consider documents as duplicates
        output_directory: Path to output directory
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store normalized text, n-grams, and signatures
    doc_text = {}
    doc_ngrams = {}
    doc_signatures = {}
    
    # 1. Process each document
    print(f"Processing {len(input_files)} documents...")
    for doc_path in input_files:
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Normalize text
            normalized_text = normalize_text(text)
            doc_text[doc_path] = text
            
            # Create n-grams
            doc_ngrams[doc_path] = create_ngrams(normalized_text, ngrams)
            
            # Compute MinHash signature
            doc_signatures[doc_path] = compute_minhash_signature(doc_ngrams[doc_path], num_hashes)
            
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not process {doc_path}: {e}")
    
    # 2. Apply LSH to find candidate duplicates
    print("Applying LSH to find candidate duplicates...")
    buckets = apply_lsh(doc_signatures, num_bands)
    
    # Filter buckets to only include those with potential duplicates (more than one document)
    candidate_buckets = {bucket: docs for bucket, docs in buckets.items() if len(docs) > 1}
    
    # 3. Find actual duplicate clusters
    print("Finding duplicate clusters...")
    duplicate_clusters = find_duplicate_clusters(candidate_buckets, doc_ngrams, jaccard_threshold)
    
    # 4. Create a set of documents to remove (all but one from each cluster)
    docs_to_remove = set()
    for cluster in duplicate_clusters:
        # Keep one random document from the cluster
        docs_to_keep = random.choice(cluster)
        
        # Mark the rest for removal
        for doc in cluster:
            if doc != docs_to_keep:
                docs_to_remove.add(doc)
    
    # 5. Write deduplicated files to output directory
    print(f"Writing {len(input_files) - len(docs_to_remove)} deduplicated files to output directory...")
    for doc_path in input_files:
        if doc_path in docs_to_remove:
            continue
            
        # Copy the document to the output directory
        input_path = Path(doc_path)
        output_path = output_dir / input_path.name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc_text[doc_path])
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not write {output_path}: {e}")
            
    print(f"Deduplication complete. Removed {len(docs_to_remove)} duplicate documents.") 