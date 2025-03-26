#!/usr/bin/env python3
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


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
