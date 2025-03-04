import numpy as np
import math
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import wave
import contextlib
import hashlib
import argparse
import glob
import pyloudnorm as pyln

# --- Function to read the audio file (supports WAV, MP3, etc.) ---
def read_audio_file(filepath):
    """
    Read an audio file (WAV, MP3, etc.) into a numpy array (mono) and return (audio_data, sample_rate).
    If the file has more than one channel, it is converted to mono by taking the average.
    For formats not directly supported by soundfile, pydub is used as a fallback.
    """
    try:
        data, fs = sf.read(filepath)
    except Exception:
        # Fallback using pydub for formats not supported by soundfile (e.g. MP3).
        audio = AudioSegment.from_file(filepath)
        fs = audio.frame_rate
        if audio.channels > 1:
            audio = audio.set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 2:  # 16-bit PCM
            data /= 2**15
        elif audio.sample_width == 3:  # 24-bit
            data /= 2**23
        elif audio.sample_width == 4:  # 32-bit
            data /= 2**31
    else:
        # If multichannel, reduce to mono.
        if data.ndim > 1:
            data = data.mean(axis=1)
    return np.array(data, dtype=np.float64), fs

# --- Function to compute the momentary loudness (0.4 s) using pyloudnorm ---
def compute_momentary_loudness(audio, fs, window_dur=0.4):
    """
    Compute the momentary loudness (0.4 s window) for the audio signal 'audio' sampled at 'fs'
    using pyloudnorm (BS.1770 K-weighting).
    
    Returns two arrays:
    - time_stamps: the center times (in seconds) of each window
    - momentary_loudness: the loudness in LUFS for each window
    """
    if len(audio) < int(window_dur * fs):
        # Audio is too short to compute a 400 ms window
        return np.array([]), np.array([])

    # Create a meter (BS.1770)
    meter = pyln.Meter(fs)

    frame_length = int(window_dur * fs)
    # We use a 100 ms hop to get a denser plot
    frame_step = int(0.1 * fs)

    time_stamps = []
    momentary_loudness = []

    num_frames = 1 + (len(audio) - frame_length) // frame_step

    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        if end > len(audio):
            break
        
        block = audio[start:end]
        # Compute loudness for the block
        block_loudness = meter.integrated_loudness(block)

        # Time at the center of the block
        time_center = (start + frame_length / 2) / fs

        time_stamps.append(time_center)
        momentary_loudness.append(block_loudness)

    return np.array(time_stamps), np.array(momentary_loudness)

# --- Function to compute the integrated loudness using pyloudnorm ---
def compute_integrated_loudness(audio, fs):
    """
    Compute the integrated loudness of the entire audio signal using pyloudnorm (BS.1770).
    Returns the integrated loudness in LUFS.
    If the audio is too short (less than 400 ms), returns -70.0 by default.
    """
    if len(audio) < int(0.4 * fs):
        return -70.0

    # Create a meter (BS.1770)
    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(audio)

    return loudness

# --- Function to create the analysis plots ---
def create_analysis_plots(audio_signal, sample_rate, times, momentary, integrated, audio_filename):
    """
    Create the analysis plots for the given audio file and return the figure:
      1) Waveform
      2) Spectrogram
      3) Spectrogram limited to 20-4000 Hz
      4) Momentary loudness (0.4 s) as a step plot + horizontal line for integrated loudness
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Loudness Analysis: {audio_filename}", fontsize=14, fontweight='bold')
    
    time_formatter = ticker.FormatStrFormatter('%.2f')
    
    # 1) Waveform
    t_audio = np.arange(len(audio_signal)) / sample_rate
    axs[0].plot(t_audio, audio_signal)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Waveform")
    axs[0].grid(True)
    axs[0].xaxis.set_major_formatter(time_formatter)
    
    # 2) Spectrogram (cmap='viridis')
    NFFT = 2048
    noverlap = 1024
    axs[1].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, noverlap=noverlap, cmap='viridis')
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Spectrogram")
    axs[1].grid(True)
    axs[1].xaxis.set_major_formatter(time_formatter)
    
    # 3) Spectrogram (cmap='magma', limited to 20-4000 Hz)
    axs[2].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, noverlap=noverlap, cmap='magma')
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_title("Spectrogram (20-4000 Hz)")
    axs[2].set_ylim(20, 4000)
    axs[2].grid(True)
    axs[2].xaxis.set_major_formatter(time_formatter)
    
    # 4) Momentary loudness step plot
    if len(times) > 0:
        axs[3].step(times, momentary, where='mid')
        axs[3].set_ylabel("Loudness (LUFS)")
        axs[3].set_title("Momentary Loudness (0.4 s)")
        axs[3].grid(True)
        axs[3].xaxis.set_major_formatter(time_formatter)
        
        # Y range
        y_min = max(-70, np.min(momentary) - 5)
        y_max = min(0, np.max(momentary) + 5)
        axs[3].set_ylim(y_min, y_max)
        
        # Horizontal line with integrated loudness
        axs[3].axhline(y=integrated, linestyle='--', alpha=0.7)
        
        # Text box with integrated loudness value
        axs[3].text(
            0.95, 0.95, f"Integrated Loudness: {integrated:.1f} LUFS",
            transform=axs[3].transAxes,
            fontsize=10, fontweight='bold', ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkred')
        )
    else:
        # If audio is too short, no plot
        axs[3].text(0.5, 0.5, "Audio too short for Momentary Loudness calculation",
                    ha='center', va='center', fontsize=10)
        axs[3].set_title("Momentary Loudness (not available)")
    
    axs[3].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, top=0.9)
    
    return fig

# --- Function to process a single audio file ---
def process_audio_file(audio_path, output_dir, create_pdf=True):
    """
    Process a single audio file by computing loudness metrics and generating output files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Read audio
        audio_signal, sample_rate = read_audio_file(audio_path)
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        
        print(f"Processing {audio_filename} ...")
        
        # Momentary loudness (0.4 s)
        times, momentary_loudness = compute_momentary_loudness(audio_signal, sample_rate)
        
        # Integrated loudness
        integrated_loudness = compute_integrated_loudness(audio_signal, sample_rate)
        
        # Compute SHA256 hash
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        print(f"  -> Integrated Loudness: {integrated_loudness:.1f} LUFS")
        print(f"  -> SHA256: {file_hash}")
        
        # Save CSV with momentary loudness
        csv_filename = os.path.join(output_dir, f"loudness_results_{audio_name}.csv")
        with open(csv_filename, "w") as f_csv:
            f_csv.write("Time_s,Momentary_LUFS\n")
            for t, val in zip(times, momentary_loudness):
                f_csv.write(f"{t:.3f},{val:.3f}\n")
        
        # Create and save plots
        fig_plots = create_analysis_plots(
            audio_signal, sample_rate,
            times, momentary_loudness,
            integrated_loudness,
            audio_filename
        )
        
        plot_filename = os.path.join(output_dir, f"loudness_chart_{audio_name}.png")
        fig_plots.savefig(plot_filename)
        plt.close(fig_plots)
        
        print(f"  -> Momentary results saved to: {csv_filename}")
        print(f"  -> Chart saved to: {plot_filename}")
        
        # PDF report
        if create_pdf:
            pdf_filename = os.path.join(output_dir, f"loudness_report_{audio_name}.pdf")
            with PdfPages(pdf_filename) as pdf:
                # A4 format
                a4_width_inch, a4_height_inch = 8.27, 11.69
                
                # Page 1: info + specs
                fig_info = plt.figure(figsize=(a4_width_inch, a4_height_inch))
                ax_header = plt.axes([0.1, 0.8, 0.8, 0.15])
                ax_header.axis('off')
                ax_info = plt.axes([0.1, 0.3, 0.8, 0.45])
                ax_info.axis('off')
                ax_footer = plt.axes([0.1, 0.05, 0.8, 0.1])
                ax_footer.axis('off')
                
                ax_header.text(0.5, 0.8, "LOUDNESS ANALYSIS REPORT", 
                               horizontalalignment='center',
                               fontsize=18, fontweight='bold')
                ax_header.text(0.5, 0.5, f"File: {audio_filename}", 
                               horizontalalignment='center',
                               fontsize=14)
                ax_header.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                frames = len(audio_signal)
                duration_s = frames / float(sample_rate)
                h = int(duration_s // 3600)
                m = int((duration_s % 3600) // 60)
                s = int(duration_s % 60)
                ms = int((duration_s - int(duration_s)) * 1000)
                formatted_duration = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
                
                # If WAV, try to get more precise info
                try:
                    with contextlib.closing(wave.open(audio_path, 'r')) as w:
                        frames_wav = w.getnframes()
                        rate_wav = w.getframerate()
                        duration_wav = frames_wav / float(rate_wav)
                        channels = w.getnchannels()
                        sampwidth = w.getsampwidth()
                        format_info = f"{channels} channel(s), {sampwidth*8} bit"
                        # Update duration
                        h = int(duration_wav // 3600)
                        m = int((duration_wav % 3600) // 60)
                        s = int(duration_wav % 60)
                        ms = int((duration_wav - int(duration_wav)) * 1000)
                        formatted_duration = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
                except:
                    channels = 1
                    format_info = "Converted to mono"
                
                analysis_params = (
                    "• Method: EBU R128 (K-weighting)\n"
                    "• Momentary Loudness: 400 ms window\n"
                    "• Integrated Loudness: gating -70 LUFS and relative -10 LU\n"
                )
                
                info_text = (
                    f"TECHNICAL DETAILS\n\n"
                    f"• File name: {audio_filename}\n"
                    f"• Format: {format_info}\n"
                    f"• Sample rate: {sample_rate} Hz\n"
                    f"• Number of samples: {frames}\n"
                    f"• Duration: {formatted_duration}\n\n"
                    f"ANALYSIS PARAMETERS\n\n{analysis_params}\n\n"
                    f"HASH\n\n"
                    f"• SHA-256: {file_hash}\n\n"
                    f"LOUDNESS ANALYSIS RESULTS\n\n"
                    f"• Integrated Loudness: {integrated_loudness:.1f} LUFS\n"
                )
                
                # Momentary stats
                if len(momentary_loudness) > 0:
                    mom_min = np.min(momentary_loudness)
                    mom_max = np.max(momentary_loudness)
                    mom_std = np.std(momentary_loudness)
                    info_text += (f"• Momentary min: {mom_min:.1f} LUFS\n"
                                  f"• Momentary max: {mom_max:.1f} LUFS\n"
                                  f"• Standard deviation: {mom_std:.1f} LUFS\n")
                
                ax_info.text(0, 1, info_text,
                             fontsize=11, verticalalignment='top',
                             horizontalalignment='left',
                             linespacing=1.5)
                
                ax_footer.text(0.5, 0.2, f"Page 1/2",
                               horizontalalignment='center',
                               fontsize=8)
                
                pdf.savefig(fig_info)
                plt.close(fig_info)
                
                # Page 2: the plot created before
                fig_plots.set_size_inches(a4_width_inch, a4_height_inch)
                # Slight title correction
                fig_plots.suptitle(f"Loudness Analysis: {audio_filename}",
                                   fontsize=12, fontweight='bold', y=0.98)
                
                # Footer
                fig_plots.text(0.5, 0.01, "Page 2/2",
                               ha='center', fontsize=8)
                
                pdf.savefig(fig_plots)
                
            print(f"  -> PDF report saved to: {pdf_filename}")
        
        return True, integrated_loudness
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False, None

# --- Main: Process the input (single file or directory) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loudness analysis for audio files using pyloudnorm (EBU R128).")
    
    parser.add_argument("input", help="Audio file or directory containing audio files to process")
    parser.add_argument("--output", help="Output directory for results", default="./output_loudness")
    parser.add_argument("--nopdf", action="store_true", help="Do not generate PDF reports")
    parser.add_argument("--file-ext", help="Filter only these file extensions (comma-separated).", default=None)
    
    args = parser.parse_args()
    
    # Common audio extensions
    common_audio_extensions = ["wav", "mp3", "ogg", "flac", "aac", "wma", "m4a", "aiff", "opus"]
    
    if args.file_ext:
        file_extensions = [ext.strip().lower() for ext in args.file_ext.split(",")]
        print(f"Processing only file extensions: {', '.join(file_extensions)}")
    else:
        file_extensions = common_audio_extensions
        print(f"Processing all common audio formats: {', '.join(file_extensions)}")
    
    if os.path.isfile(args.input):
        # Single file
        success, loudness = process_audio_file(args.input, args.output, not args.nopdf)
        if success:
            print(f"Processing completed. Integrated Loudness: {loudness:.1f} LUFS")
        else:
            print("Error during processing.")
    
    elif os.path.isdir(args.input):
        # Directory
        import glob
        patterns = [os.path.join(args.input, f"*.{ext}") for ext in file_extensions]
        audio_files = []
        for pattern in patterns:
            audio_files.extend(glob.glob(pattern))
        
        if not audio_files:
            print(f"No audio files found in {args.input} with extensions {file_extensions}")
            exit(1)
        
        print(f"Found {len(audio_files)} audio files to process.")
        
        # Summary CSV
        summary_csv = os.path.join(args.output, "loudness_summary.csv")
        os.makedirs(args.output, exist_ok=True)
        
        with open(summary_csv, "w") as f:
            f.write("Filename,Integrated_LUFS,Success\n")
            
            for audio_file in audio_files:
                filename = os.path.basename(audio_file)
                print(f"\nProcessing {filename}...")
                
                success, loudness = process_audio_file(audio_file, args.output, not args.nopdf)
                
                if success and loudness is not None:
                    f.write(f"{filename},{loudness:.1f},1\n")
                    print(f"  -> Integrated Loudness: {loudness:.1f} LUFS")
                else:
                    f.write(f"{filename},N/A,0\n")
                    print("  -> Loudness calculation error")
        
        print(f"\nSummary saved to: {summary_csv}")
    
    else:
        print(f"Error: '{args.input}' is not a valid file or directory.")
        exit(1)
    
    print("\nProcessing complete.")
