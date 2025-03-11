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

# --- Function to format time as HH:MM:SS.mmm (added from fstic.py) ---
def format_time_hhmmssmmm(seconds_float):
    """
    Convert a floating-point 'seconds_float' into a time string HH:MM:SS.mmm
    where HH, MM, SS are zero-padded and mmm = milliseconds.
    """
    hours = int(seconds_float // 3600)
    remainder = seconds_float % 3600
    minutes = int(remainder // 60)
    seconds = remainder % 60
    millis = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

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

# --- Function to create the analysis plots (modified to match fstic style) ---
def create_analysis_plots(audio_signal, sample_rate, times, momentary, integrated, audio_filename):
    """
    Create the analysis plots for the given audio file and return the figure:
      1) Waveform
      2) Spectrogram
      3) Spectrogram limited to 20-4000 Hz
      4) Momentary loudness (0.4 s) as a step plot + horizontal line for integrated loudness
    """
    # Calculate total duration
    total_duration = len(audio_signal) / sample_rate
    mid_time = total_duration / 2.0
    
    fig, axs = plt.subplots(4, 1, figsize=(9, 10))  # Reduced width for better margins
    
    # 1) Waveform
    t_audio = np.arange(len(audio_signal)) / sample_rate
    axs[0].plot(t_audio, audio_signal)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Waveform", pad=10)
    axs[0].grid(True)
    
    # 2) Spectrogram (cmap='viridis')
    NFFT = 2048
    noverlap = 1024
    axs[1].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, noverlap=noverlap, cmap='viridis')
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Spectrogram", pad=10)
    axs[1].grid(True)
    
    # 3) Spectrogram (cmap='magma', limited to 20-4000 Hz)
    axs[2].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, noverlap=noverlap, cmap='magma')
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_title("Spectrogram (20-4000 Hz)", pad=10)
    axs[2].set_ylim(20, 4000)
    axs[2].grid(True)
    
    # 4) Momentary loudness step plot
    if len(times) > 0:
        axs[3].step(times, momentary, where='post', color='blue')
        axs[3].set_ylabel("Loudness (LUFS)")
        axs[3].set_title("Momentary Loudness (0.4 s)", pad=10)
        axs[3].grid(True)
        
        # Y range
        y_min = max(-70, np.min(momentary) - 5)
        y_max = min(0, np.max(momentary) + 5)
        axs[3].set_ylim(y_min, y_max)
        
        # Horizontal line with integrated loudness
        axs[3].axhline(y=integrated, color='darkblue', linestyle='--', alpha=0.7)
        
        # Text box with integrated loudness value
        axs[3].text(
            0.95, 0.95, f"Integrated Loudness: {integrated:.1f} LUFS",
            transform=axs[3].transAxes,
            fontsize=10, fontweight='bold', ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkblue', boxstyle='round,pad=0.5')
        )
    else:
        # If audio is too short, no plot
        axs[3].text(0.5, 0.5, "Audio too short for Momentary Loudness calculation",
                    ha='center', va='center', fontsize=10)
        axs[3].set_title("Momentary Loudness (not available)")
    
    # Set x-ticks and x-labels for each subplot (similar to fstic.py)
    for ax in axs:
        ax.set_xticks([0, mid_time, total_duration])
        ax.set_xticklabels([
            format_time_hhmmssmmm(0),
            format_time_hhmmssmmm(mid_time),
            format_time_hhmmssmmm(total_duration)
        ])
        ax.set_xlim([0, total_duration])
        ax.set_xlabel("Time")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, top=0.95, left=0.15, right=0.92)
    
    return fig

# --- Comparison plots for two files side by side (like in fstic.py) ---
def create_comparison_plots(
    audio_signal1, fs1, times1, momentary1, integrated1, name1,
    audio_signal2, fs2, times2, momentary2, integrated2, name2
):
    """
    Creates a figure with 4 rows and 2 columns, comparing:
      Row 1: Waveform (File1 left, File2 right)
      Row 2: Spectrogram (File1 left, File2 right)
      Row 3: Spectrogram (limited freq) (File1 left, File2 right)
      Row 4: Momentary loudness (File1 left, File2 right)
    
    Only 3 x-axis ticks (start, mid, end) for each column.
    """
    total_duration1 = len(audio_signal1) / fs1
    mid_time1 = total_duration1 / 2.0
    
    total_duration2 = len(audio_signal2) / fs2
    mid_time2 = total_duration2 / 2.0
    
    fig, axs = plt.subplots(4, 2, figsize=(10.5, 12))
    
    NFFT = 2048
    noverlap = 1024
    
    # Row 1: Waveforms
    t1 = np.arange(len(audio_signal1)) / fs1
    t2 = np.arange(len(audio_signal2)) / fs2
    
    # File1 waveform (left)
    axs[0, 0].plot(t1, audio_signal1)
    axs[0, 0].set_title(f"Waveform\n{name1}", pad=10)
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True)
    
    # File2 waveform (right)
    axs[0, 1].plot(t2, audio_signal2)
    axs[0, 1].set_title(f"Waveform\n{name2}", pad=10)
    axs[0, 1].grid(True)
    
    # Row 2: Spectrogram (File1 left, File2 right)
    axs[1, 0].specgram(audio_signal1, NFFT=NFFT, Fs=fs1, noverlap=noverlap, cmap='viridis')
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_title(f"Spectrogram\n{name1}", pad=10)
    axs[1, 0].grid(True)
    
    axs[1, 1].specgram(audio_signal2, NFFT=NFFT, Fs=fs2, noverlap=noverlap, cmap='viridis')
    axs[1, 1].set_title(f"Spectrogram\n{name2}", pad=10)
    axs[1, 1].grid(True)
    
    # Row 3: Spectrogram limited 20-4000 Hz
    axs[2, 0].specgram(audio_signal1, NFFT=NFFT, Fs=fs1, noverlap=noverlap, cmap='magma')
    axs[2, 0].set_ylabel("Frequency (Hz)")
    axs[2, 0].set_ylim(20, 4000)
    axs[2, 0].set_title(f"Spectrogram (20-4000 Hz)\n{name1}", pad=10)
    axs[2, 0].grid(True)
    
    axs[2, 1].specgram(audio_signal2, NFFT=NFFT, Fs=fs2, noverlap=noverlap, cmap='magma')
    axs[2, 1].set_ylim(20, 4000)
    axs[2, 1].set_title(f"Spectrogram (20-4000 Hz)\n{name2}", pad=10)
    axs[2, 1].grid(True)
    
    # Row 4: Momentary loudness (File1 left, File2 right)
    if len(times1) > 0:
        axs[3, 0].step(times1, momentary1, where='post', color='blue')
        axs[3, 0].axhline(y=integrated1, color='darkblue', linestyle='--', alpha=0.7)
        # Y range
        y_min1 = max(-70, np.min(momentary1) - 5)
        y_max1 = min(0, np.max(momentary1) + 5)
        axs[3, 0].set_ylim(y_min1, y_max1)
        axs[3, 0].set_ylabel("Loudness (LUFS)")
        axs[3, 0].set_title(f"Momentary Loudness\n{name1}", pad=10)
        axs[3, 0].grid(True)
        
        # Add text box with integrated loudness
        axs[3, 0].text(
            0.95, 0.95, f'Integrated: {integrated1:.1f} LUFS', 
            transform=axs[3, 0].transAxes,
            color='darkblue', fontsize=10, fontweight='bold',
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkblue', boxstyle='round,pad=0.5')
        )
    else:
        axs[3, 0].text(0.5, 0.5, "Audio too short for Momentary Loudness",
                    ha='center', va='center', fontsize=10)
        axs[3, 0].set_title(f"Momentary Loudness (not available)\n{name1}", pad=10)
    
    if len(times2) > 0:
        axs[3, 1].step(times2, momentary2, where='post', color='blue')
        axs[3, 1].axhline(y=integrated2, color='darkblue', linestyle='--', alpha=0.7)
        # Y range
        y_min2 = max(-70, np.min(momentary2) - 5)
        y_max2 = min(0, np.max(momentary2) + 5)
        axs[3, 1].set_ylim(y_min2, y_max2)
        axs[3, 1].set_title(f"Momentary Loudness\n{name2}", pad=10)
        axs[3, 1].grid(True)
        
        # Add text box with integrated loudness
        axs[3, 1].text(
            0.95, 0.95, f'Integrated: {integrated2:.1f} LUFS', 
            transform=axs[3, 1].transAxes,
            color='darkblue', fontsize=10, fontweight='bold',
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkblue', boxstyle='round,pad=0.5')
        )
    else:
        axs[3, 1].text(0.5, 0.5, "Audio too short for Momentary Loudness",
                    ha='center', va='center', fontsize=10)
        axs[3, 1].set_title(f"Momentary Loudness (not available)\n{name2}", pad=10)
    
    # Set x-ticks for each column in each row
    # Left column -> file1
    for row in range(4):
        axs[row, 0].set_xticks([0, mid_time1, total_duration1])
        axs[row, 0].set_xticklabels([
            format_time_hhmmssmmm(0),
            format_time_hhmmssmmm(mid_time1),
            format_time_hhmmssmmm(total_duration1)
        ])
        axs[row, 0].set_xlim([0, total_duration1])
        axs[row, 0].set_xlabel("Time")
    
    # Right column -> file2
    for row in range(4):
        axs[row, 1].set_xticks([0, mid_time2, total_duration2])
        axs[row, 1].set_xticklabels([
            format_time_hhmmssmmm(0),
            format_time_hhmmssmmm(mid_time2),
            format_time_hhmmssmmm(total_duration2)
        ])
        axs[row, 1].set_xlim([0, total_duration2])
        axs[row, 1].set_xlabel("Time")
    
    # Adjust spacing to reduce overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.5, top=0.95, left=0.15, right=0.92)
    return fig

# --- Function to process a single audio file (updated for new PDF style) ---
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
            f_csv.write("Time,Momentary_LUFS\n")
            for t, val in zip(times, momentary_loudness):
                t_formatted = format_time_hhmmssmmm(t)
                f_csv.write(f"{t_formatted},{val:.3f}\n")
        
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
                dur_str = format_time_hhmmssmmm(duration_s)
                
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
                        dur_str = format_time_hhmmssmmm(duration_wav)
                        frames = frames_wav
                        sample_rate = rate_wav
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
                    f"• Number of samples: {frames:,}\n"
                    f"• Duration: {dur_str}\n\n"
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
                fig_plots = create_analysis_plots(
                    audio_signal, sample_rate,
                    times, momentary_loudness,
                    integrated_loudness,
                    audio_filename
                )
                fig_plots.set_size_inches(a4_width_inch, a4_height_inch)
                
                # Adjust the figure margins for PDF output
                plt.subplots_adjust(left=0.15, right=0.90, hspace=0.8, top=0.95)
                
                # Footer
                fig_plots.text(0.5, 0.01, "Page 2/2",
                               ha='center', fontsize=8)
                
                pdf.savefig(fig_plots)
                plt.close(fig_plots)
                
            print(f"  -> PDF report saved to: {pdf_filename}")
        
        return True, integrated_loudness
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False, None

# --- Compare two audio files side by side ---
def compare_two_audio_files(file1, file2, output_dir, create_pdf=True):
    """
    Computes loudness for two audio files and generates a side-by-side comparison plot and PDF.
    Also creates a combined CSV with times in HH:MM:SS.mmm format, using the actual filenames in headers.
    
    Returns (success, loudness1, loudness2)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Read both files
        audio_signal1, fs1 = read_audio_file(file1)
        audio_signal2, fs2 = read_audio_file(file2)
        filename1 = os.path.basename(file1)  # keep extension as user requested
        filename2 = os.path.basename(file2)
        
        name1 = os.path.splitext(filename1)[0]  # used for some titles
        name2 = os.path.splitext(filename2)[0]
        
        print(f"Comparing:\n  File A: {file1}\n  File B: {file2}")
        
        # Compute loudness for both files
        times1, momentary1 = compute_momentary_loudness(audio_signal1, fs1)
        times2, momentary2 = compute_momentary_loudness(audio_signal2, fs2)
        
        integrated1 = compute_integrated_loudness(audio_signal1, fs1)
        integrated2 = compute_integrated_loudness(audio_signal2, fs2)
        
        # Calculate loudness statistics
        mom1_min = np.min(momentary1) if len(momentary1) > 0 else -70
        mom1_max = np.max(momentary1) if len(momentary1) > 0 else -70
        mom1_std = np.std(momentary1) if len(momentary1) > 0 else 0
        
        mom2_min = np.min(momentary2) if len(momentary2) > 0 else -70
        mom2_max = np.max(momentary2) if len(momentary2) > 0 else -70
        mom2_std = np.std(momentary2) if len(momentary2) > 0 else 0
        
        # Calculate SHA256 for both
        with open(file1, 'rb') as f:
            hash1 = hashlib.sha256(f.read()).hexdigest()
        with open(file2, 'rb') as f:
            hash2 = hashlib.sha256(f.read()).hexdigest()
        
        print(f"File A Integrated Loudness: {integrated1:.1f} LUFS")
        print(f"File B Integrated Loudness: {integrated2:.1f} LUFS")
        
        # Create combined CSV for comparison
        csv_filename = os.path.join(output_dir, f"loudness_comparison_{name1}_vs_{name2}.csv")
        
        max_len = max(len(times1), len(times2))
        
        # Use underscores for any spaces in filenames to keep CSV simpler
        header_timeA = f"Time_{filename1.replace(' ', '_')}"
        header_loudnessA = f"Loudness_{filename1.replace(' ', '_')}"
        header_timeB = f"Time_{filename2.replace(' ', '_')}"
        header_loudnessB = f"Loudness_{filename2.replace(' ', '_')}"
        
        with open(csv_filename, "w") as f:
            f.write(f"{header_timeA},{header_loudnessA},{header_timeB},{header_loudnessB}\n")
            for i in range(max_len):
                if i < len(times1):
                    tA_str = format_time_hhmmssmmm(times1[i])
                    loudA_str = f"{momentary1[i]:.3f}"
                else:
                    tA_str = ""
                    loudA_str = ""
                
                if i < len(times2):
                    tB_str = format_time_hhmmssmmm(times2[i])
                    loudB_str = f"{momentary2[i]:.3f}"
                else:
                    tB_str = ""
                    loudB_str = ""
                
                f.write(f"{tA_str},{loudA_str},{tB_str},{loudB_str}\n")
        
        print(f"Comparison CSV saved to {csv_filename}")
        
        # Create side-by-side comparison plots
        fig_compare = create_comparison_plots(
            audio_signal1, fs1, times1, momentary1, integrated1, name1,
            audio_signal2, fs2, times2, momentary2, integrated2, name2
        )
        
        plot_filename = os.path.join(output_dir, f"chart_comparison_{name1}_vs_{name2}.png")
        fig_compare.savefig(plot_filename)
        plt.close(fig_compare)
        print(f"Comparison plot saved to {plot_filename}")
        
        if create_pdf:
            pdf_filename = os.path.join(output_dir, f"report_comparison_{name1}_vs_{name2}.pdf")
            with PdfPages(pdf_filename) as pdf:
                a4_width_inch, a4_height_inch = 8.27, 11.69
                
                # First page: textual summary for both files
                fig_info = plt.figure(figsize=(a4_width_inch, a4_height_inch))
                ax_header = plt.axes([0.1, 0.8, 0.8, 0.15])
                ax_header.axis('off')
                ax_info = plt.axes([0.1, 0.2, 0.8, 0.55])
                ax_info.axis('off')
                ax_footer = plt.axes([0.1, 0.05, 0.8, 0.1])
                ax_footer.axis('off')
                
                ax_header.text(0.5, 0.8, "LOUDNESS COMPARISON REPORT", 
                               horizontalalignment='center', fontsize=18, fontweight='bold')
                ax_header.text(0.5, 0.5, f"Files: {filename1} vs {filename2}", 
                               horizontalalignment='center', fontsize=14)
                ax_header.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                # Info about File A
                framesA = len(audio_signal1)
                durA_sec = framesA / fs1 if fs1 > 0 else 0
                durA = format_time_hhmmssmmm(durA_sec)
                try:
                    with contextlib.closing(wave.open(file1, 'r')) as wfA:
                        chA = wfA.getnchannels()
                        swA = wfA.getsampwidth()
                        infoA = f"{chA} channels, {swA*8} bit"
                except:
                    infoA = "Converted/Unknown (Mono)"
                
                # Info about File B
                framesB = len(audio_signal2)
                durB_sec = framesB / fs2 if fs2 > 0 else 0
                durB = format_time_hhmmssmmm(durB_sec)
                try:
                    with contextlib.closing(wave.open(file2, 'r')) as wfB:
                        chB = wfB.getnchannels()
                        swB = wfB.getsampwidth()
                        infoB = f"{chB} channels, {swB*8} bit"
                except:
                    infoB = "Converted/Unknown (Mono)"
                
                info_text = (
                    f"FILE A: {filename1}\n"
                    f"  • Format: {infoA}\n"
                    f"  • Sampling rate: {fs1} Hz\n"
                    f"  • Number of samples: {framesA:,}\n"
                    f"  • Duration: {durA}\n"
                    f"  • SHA-256: {hash1}\n"
                    f"  • Integrated Loudness: {integrated1:.1f} LUFS\n"
                    f"  • Momentary min: {mom1_min:.1f} LUFS\n"
                    f"  • Momentary max: {mom1_max:.1f} LUFS\n"
                    f"  • Standard deviation: {mom1_std:.1f} LUFS\n\n"
                    
                    f"FILE B: {filename2}\n"
                    f"  • Format: {infoB}\n"
                    f"  • Sampling rate: {fs2} Hz\n"
                    f"  • Number of samples: {framesB:,}\n"
                    f"  • Duration: {durB}\n"
                    f"  • SHA-256: {hash2}\n"
                    f"  • Integrated Loudness: {integrated2:.1f} LUFS\n"
                    f"  • Momentary min: {mom2_min:.1f} LUFS\n"
                    f"  • Momentary max: {mom2_max:.1f} LUFS\n"
                    f"  • Standard deviation: {mom2_std:.1f} LUFS\n\n"
                    
                    f"ANALYSIS PARAMETERS\n"
                    f"  • Method: EBU R128 (K-weighting)\n"
                    f"  • Momentary Loudness: 400 ms window\n"
                    f"  • Integrated Loudness: gating -70 LUFS and relative -10 LU\n"
                )
                
                ax_info.text(0, 1, info_text, fontsize=11, verticalalignment='top', 
                             horizontalalignment='left', linespacing=1.5)
                ax_footer.text(0.5, 0.2, "Page 1/2", horizontalalignment='center', fontsize=8)
                
                pdf.savefig(fig_info)
                plt.close(fig_info)
                
                # Second page: the comparison figure
                # We'll recreate the comparison figure since we've closed it
                fig_compare = create_comparison_plots(
                    audio_signal1, fs1, times1, momentary1, integrated1, name1,
                    audio_signal2, fs2, times2, momentary2, integrated2, name2
                )
                fig_compare.set_size_inches(a4_width_inch, a4_height_inch)
                
                # Adjust the figure margins for PDF output
                plt.subplots_adjust(left=0.15, right=0.90, hspace=0.8, wspace=0.5, top=0.95)
                
                fig_compare.text(0.5, 0.01, "Page 2/2", ha='center', fontsize=8)
                pdf.savefig(fig_compare)
                plt.close(fig_compare)
            
            print(f"PDF comparison report saved to {pdf_filename}")
        
        return True, integrated1, integrated2
    except Exception as e:
        print(f"Error while comparing files:\n  {file1}\n  {file2}\n  Error: {e}")
        return False, None, None
    

# --- Main: Process the input (single file or directory or compare two files) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loudness analysis for audio files using pyloudnorm (EBU R128).")
    
    # Positional/optional
    parser.add_argument("input", nargs="?", help="Audio file or folder. Not used when --compare is set.", default=None)
    
    parser.add_argument("--output", help="Output directory for results", default="./output_loudness")
    parser.add_argument("--nopdf", action="store_true", help="Do not generate PDF reports")
    parser.add_argument("--file-ext", help="Filter only these file extensions (comma-separated).", default=None)
    
    # New parameter for 2-file comparison
    parser.add_argument("--compare", nargs=2, help="Compare two audio files side by side. Usage: --compare fileA fileB")
    
    args = parser.parse_args()
    
    # Common audio extensions
    common_audio_extensions = ["wav", "mp3", "ogg", "flac", "aac", "wma", "m4a", "aiff", "opus"]
    
    if args.file_ext:
        file_extensions = [ext.strip().lower() for ext in args.file_ext.split(",")]
        print(f"Processing only file extensions: {', '.join(file_extensions)}")
    else:
        file_extensions = common_audio_extensions
        print(f"Processing all common audio formats: {', '.join(file_extensions)}")
    
    if args.compare:
        # Comparison mode
        fileA, fileB = args.compare
        if not (os.path.isfile(fileA) and os.path.isfile(fileB)):
            print("Error: --compare requires two valid files.")
            exit(1)
        
        success, loudnessA, loudnessB = compare_two_audio_files(
            fileA, fileB,
            args.output,
            create_pdf=(not args.nopdf)
        )
        if success:
            print(f"Comparison completed. Loudness {os.path.basename(fileA)}={loudnessA:.1f} LUFS, {os.path.basename(fileB)}={loudnessB:.1f} LUFS")
        else:
            print("Comparison failed.")
        exit(0)
    
    # If not in compare mode, check the 'input'
    if args.input is None:
        print("Error: you must specify either an input file/folder or use --compare")
        exit(1)
    
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