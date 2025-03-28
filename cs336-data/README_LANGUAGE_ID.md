# Language Identification Analysis

This directory contains scripts for analyzing languages in web pages from WARC files.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Analysis

### Using Existing WARC Files

You can run the analysis on your existing WARC files in several ways:

1. **Automatically find and use WARC files in the current directory**:
   ```
   python download_and_analyze_warc.py
   ```

2. **Specify a particular WARC file to analyze**:
   ```
   python download_and_analyze_warc.py --file path/to/your/file.warc.gz
   ```

3. **Search for WARC files in a specific directory**:
   ```
   python download_and_analyze_warc.py --dir path/to/directory/with/warc/files
   ```

4. **Customize the number of samples to analyze**:
   ```
   python download_and_analyze_warc.py --samples 30
   ```

You can also run the language identification script directly:

```
python run_language_identification.py path/to/your/warcfile.warc.gz
```

With custom number of samples:

```
python run_language_identification.py path/to/your/warcfile.warc.gz --samples 30
```

## Interpreting the Results

The script displays:
1. For each sample:
   - The detected language code
   - The confidence score
   - The URL of the web page
   - A text snippet to verify the language

2. Summary statistics:
   - Distribution of detected languages
   - Percentage of English documents
   - Average confidence score

Use these results to:
- Verify the accuracy of the language detection
- Determine an appropriate confidence threshold for filtering
- Understand the language distribution in the dataset 