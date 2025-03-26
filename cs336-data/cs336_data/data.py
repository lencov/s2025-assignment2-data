#!/usr/bin/env python3
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import os
import re
from pathlib import Path


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    Extract plain text from a byte string containing HTML.
    
    Args:
        html_bytes: Byte string containing HTML data
        
    Returns:
        Extracted plain text as a string
    """
    # Try to decode as UTF-8 first
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, detect the encoding and try again
        detected_encoding = detect_encoding(html_bytes)
        try:
            html_str = html_bytes.decode(detected_encoding)
        except (UnicodeDecodeError, LookupError):
            # Fall back to latin-1 as it can decode any byte string
            html_str = html_bytes.decode('latin-1')
    
    # Extract plain text from the HTML
    text = extract_plain_text(html_str)
    return text


def identify_language(text: str) -> tuple[str, float]:
    """
    Identify the language of a given text using character set detection.
    
    Args:
        text: Text string to identify language for
        
    Returns:
        A tuple containing (language_code, confidence_score)
        where language_code is a string (e.g., "en" for English)
        and confidence_score is a float between 0 and 1
    """
    # Ensure text is not empty
    if not text or text.isspace():
        return "und", 0.0  # Return "undefined" with 0 confidence for empty text
    
    # Simple language detection for Chinese
    # If text contains significant Chinese characters, classify as Chinese
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chinese_chars) > 5 or (len(chinese_chars) > 0 and len(chinese_chars) / len(text) > 0.1):
        return "zh", 0.9
    
    # Simple detection for English and other Latin-based languages
    # Check for English by looking at common English words
    english_words = ["the", "and", "of", "to", "a", "in", "that", "is", "was", "for", "with", "on", "as", "by", "at"]
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count English words
    english_word_count = sum(1 for word in words if word in english_words)
    
    # If sufficient English words are found, classify as English
    if english_word_count > 3 or (words and english_word_count / len(words) > 0.1):
        return "en", 0.8
    
    # If we still can't determine the language but it uses Latin characters, default to English
    latin_chars = re.findall(r'[a-zA-Z]', text)
    if latin_chars and len(latin_chars) / len(text) > 0.5:
        return "en", 0.6
    
    # If unknown, return "und" with low confidence
    return "und", 0.3
