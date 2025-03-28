# PII Masking Analysis

This directory contains scripts for analyzing personally identifiable information (PII) in web pages from WARC files.

## Types of PII Detected

The script can detect and mask three types of PII:

1. **Email Addresses**: Replaces email addresses with `|||EMAIL_ADDRESS|||`
2. **Phone Numbers**: Replaces phone numbers with `|||PHONE_NUMBER|||`
3. **IP Addresses**: Replaces IP addresses with `|||IP_ADDRESS|||`

## Running the Analysis

### Option 1: Using the Helper Script

The easiest way to run the analysis is to use the helper script:

```
python download_and_analyze_warc.py --analysis pii
```

Additional options:

```
python download_and_analyze_warc.py --analysis pii --file path/to/your/file.warc.gz
```

```
python download_and_analyze_warc.py --analysis pii --dir path/to/directory/with/warc/files
```

```
python download_and_analyze_warc.py --analysis pii --samples 100
```

### Option 2: Running the PII Analysis Script Directly

You can also run the PII masking analysis script directly:

```
python run_pii_masking.py path/to/your/warcfile.warc.gz
```

With custom number of samples:

```
python run_pii_masking.py path/to/your/warcfile.warc.gz --samples 100
```

Limiting the number of displayed examples per PII type:

```
python run_pii_masking.py path/to/your/warcfile.warc.gz --examples 10
```

## Interpreting the Results

The script displays:

1. For each type of PII (emails, phones, IPs):
   - The total number found
   - Up to 20 random examples where PII was masked
   - For each example:
     - The URL of the web page
     - The original PII that was masked
     - The context before and after masking

2. Analysis notes to help identify:
   - False positives: Items that were masked but are not actually PII
   - False negatives: PII that was not masked but should have been

## Examining False Positives and Negatives

When reviewing the results, look for:

- **False Positives**: 
  - Email: Text with '@' that isn't an email (e.g., Twitter handles)
  - Phone: Numbers that look like phone numbers but aren't (e.g., product codes)
  - IP: Number sequences that match the IP format but aren't IPs (e.g., version numbers)

- **False Negatives** (harder to spot):
  - Look at the context of the masked PII
  - Check if there might be other PII nearby that wasn't detected
  - Look for unusual formats of emails, phones, or IPs that our regexes didn't catch 