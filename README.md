# FLOUC - Forensic LOUdness Calculator

![FLOUC Banner](https://img.shields.io/badge/FLOUC-Forensic%20LOUdness%20Calculator-blue)
![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)

FLOUC is a professional-grade audio loudness analysis tool designed for forensic audio analysis and audio engineering applications. It calculates and visualizes audio loudness metrics according to the EBU R128 / ITU-R BS.1770 standard using the `pyloudnorm` implementation.

## Features

- **EBU R128 / ITU-R BS.1770 Compliant**: Industry-standard loudness measurements
- **Multi-format Support**: Process WAV, MP3, FLAC, OGG, and more
- **Comprehensive Analysis**: Calculate integrated loudness and momentary loudness
- **Rich Visualizations**: Generate waveforms, spectrograms, and loudness graphs
- **Detailed Reports**: Output to CSV files and comprehensive PDF reports
- **Batch Processing**: Process individual files or entire directories
- **File Comparison**: Compare two audio files side by side with visual and numerical analysis
- **File Integrity**: SHA-256 hash verification for each analyzed file
- **Forensic Ready**: Designed with audio forensics principles in mind

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/FLOUC.git
   cd FLOUC
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python flouc.py path/to/audio_file.wav
```

This will:
- Analyze the audio file
- Generate visualizations
- Create a PDF report
- Output CSV data files
- Display a summary of loudness metrics

### Compare Two Audio Files

```bash
python flouc.py --compare path/to/file1.wav path/to/file2.wav
```

This will:
- Analyze both audio files
- Generate side-by-side comparison visualizations
- Create a comparative PDF report
- Output a combined CSV with data from both files
- Display summary metrics for both files

### Command Line Options

```bash
python flouc.py path/to/file_or_directory [--output OUTPUT_DIR] [--nopdf] [--file-ext wav,mp3]
python flouc.py --compare file1 file2 [--output OUTPUT_DIR] [--nopdf]
```

Parameters:
- `input`: Path to audio file or directory containing audio files (not used with --compare)
- `--compare`: Compare two audio files side by side (requires exactly two file paths)
- `--output`: Output directory for results (default: ./output_loudness)
- `--nopdf`: Skip PDF report generation
- `--file-ext`: Filter only specified file extensions (comma-separated)

### Batch Processing

To process all supported audio files in a directory:

```bash
python flouc.py path/to/directory
```

This will generate a summary CSV file along with individual reports for each audio file.

## Output Files

### For Individual Files

For each processed audio file, FLOUC generates:

1. **CSV Data**: `loudness_results_filename.csv` - Contains time-aligned momentary loudness values
2. **Chart**: `loudness_chart_filename.png` - Visual representation of all analysis
3. **PDF Report**: `loudness_report_filename.pdf` - Comprehensive two-page report with:
   - Technical details (file info, duration, format)
   - Loudness metrics (integrated and momentary)
   - SHA-256 hash for file integrity verification
   - Visualizations (waveform, spectrograms, loudness graph)
4. **Summary**: `loudness_summary.csv` - When processing directories, includes all files' metrics

### For File Comparisons

When comparing two files, FLOUC generates:

1. **Comparison CSV**: `loudness_comparison_file1_vs_file2.csv` - Combined CSV with data from both files
2. **Comparison Chart**: `chart_comparison_file1_vs_file2.png` - Side-by-side visual comparison
3. **Comparison PDF**: `report_comparison_file1_vs_file2.pdf` - Two-page report with:
   - Technical details for both files
   - Comparative loudness metrics
   - Side-by-side visualizations (waveforms, spectrograms, loudness)

## Core Functions

### Audio Processing

- `read_audio_file(filepath)`: Reads various audio formats into a numpy array, converting to mono if needed
- `compute_momentary_loudness(audio, fs, window_dur=0.4)`: Calculates momentary loudness using 400ms windows
- `compute_integrated_loudness(audio, fs)`: Calculates integrated loudness for the entire audio file

### Visualization

- `create_analysis_plots(...)`: Generates a multi-panel figure with:
  - Waveform visualization
  - Full frequency range spectrogram
  - Limited range spectrogram (20-4000 Hz)
  - Momentary loudness step plot with integrated loudness reference line
- `create_comparison_plots(...)`: Generates side-by-side comparison of all visualizations for two files

### File Handling

- `process_audio_file(audio_path, output_dir, create_pdf=True)`: Processes a single audio file
- `compare_two_audio_files(file1, file2, output_dir, create_pdf=True)`: Processes and compares two audio files
- Main function: Handles command-line arguments, batch processing, and comparison mode

## Technical Background

FLOUC implements the EBU R128 / ITU-R BS.1770 loudness measurement standard, which defines:

- **Integrated Loudness**: Overall loudness of the content
- **Momentary Loudness**: Short-term loudness calculated over 400ms windows
- **K-weighting**: A frequency weighting that reflects how humans perceive loudness

The implementation uses the `pyloudnorm` library, which provides accurate loudness measurements according to these standards.

## Dependencies

See [requirements.txt](requirements.txt) for the complete list of dependencies.

Main dependencies:
- `numpy`: Numerical processing
- `soundfile`: Audio file I/O
- `pydub`: Additional audio format support
- `matplotlib`: Visualization
- `pyloudnorm`: EBU R128 loudness calculations

## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) - see the LICENSE file for details.

If you use this software please cite this repo: [FLOUC - Forensic LOUdness Calculator (https://github.com/StefFriend/FLOUC)](https://github.com/StefFriend/FLOUC)

## Acknowledgments

- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) - Christian Steinmetz's implementation of ITU-R BS.1770
- EBU R128 and ITU-R BS.1770 standards for loudness measurement
