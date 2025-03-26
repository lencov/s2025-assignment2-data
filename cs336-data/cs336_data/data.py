#!/usr/bin/env python3
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import os
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
    Identify the language of a given text using fastText.
    
    Args:
        text: Text string to identify language for
        
    Returns:
        A tuple containing (language_code, confidence_score)
        where language_code is a string (e.g., "en" for English)
        and confidence_score is a float between 0 and 1
    """
    # Path to the fastText model
    # Look for model in common locations
    possible_paths = [
        "lid.176.bin",                                # Current directory
        os.path.expanduser("~/lid.176.bin"),          # Home directory
        os.path.expanduser("~/.fasttext/lid.176.bin") # Hidden directory in home
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            "Language identification model not found. Please download it from "
            "https://fasttext.cc/docs/en/language-identification.html and "
            "place it in the current directory or your home directory."
        )
    
    # Load the model
    model = fasttext.load_model(model_path)
    
    # Ensure text is not empty
    if not text or text.isspace():
        return "und", 0.0  # Return "undefined" with 0 confidence for empty text
    
    # Get prediction
    # fastText requires text to be on a single line
    text = text.replace('\n', ' ')
    predictions = model.predict(text, k=1)
    
    # Extract language code and probability
    language_code = predictions[0][0].replace('__label__', '')
    confidence_score = float(predictions[1][0])
    
    return language_code, confidence_score
