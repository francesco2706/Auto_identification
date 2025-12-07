import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------
# Funzione: genera segmenti da 5 secondi
# ---------------------------------------
def get_5sec_segments(y, sr, segment_sec=5):
    segment_samples = sr * segment_sec
    segments = []

    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        if end <= len(y):
            segments.append(y[start:end])

    return segments


# ---------------------------------------
# Funzione per generare spettrogrammi
# ---------------------------------------
def create_spectrograms(audio_path, output_dir, file_prefix, segment_sec=5):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Impossibile caricare {audio_path}: {e}")
        return

    segments = get_5sec_segments(y, sr, segment_sec)

    if len(segments) == 0:
        print(f"Audio troppo breve {audio_path}, saltato.")
        return

    for i, y_segment in enumerate(segments):
        D = librosa.stft(y_segment)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig, ax = plt.subplots(figsize=(4,4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)

        save_name = f"{file_prefix}_part{i+1}.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)


# ---------------------------------------
# Funzione per generare waveform images
# ---------------------------------------
def create_waveforms(audio_path, output_dir, file_prefix, segment_sec=5):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Impossibile caricare {audio_path}: {e}")
        return

    segments = get_5sec_segments(y, sr, segment_sec)

    if len(segments) == 0:
        print(f"Audio troppo breve {audio_path}, saltato.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, y_segment in enumerate(segments):

        # Normalizzazione
        y_segment = y_segment / (np.max(np.abs(y_segment)) + 1e-9)

        # Figura stile "waveshow" pulito
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # waveform blu identica a waveshow
        librosa.display.waveshow(y_segment, sr=sr, color="#1f77b4", linewidth=0.7, ax=ax)

        # rimuovo tutto
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

        save_name = f"{file_prefix}_part{i+1}.png"
        save_path = os.path.join(output_dir, save_name)

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


# ---------------------------------------
# Funzione batch process
# ---------------------------------------
def batch_process(input_root_folder, output_root_folder_spectrogram, output_root_folder_waveform, segment_sec=5):
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')

    for root, dirs, files in os.walk(input_root_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):

                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_root_folder)

                # Creazione cartelle output
                output_dir_spec = os.path.join(output_root_folder_spectrogram, rel_path)
                output_dir_wave = os.path.join(output_root_folder_waveform, rel_path)
                os.makedirs(output_dir_spec, exist_ok=True)
                os.makedirs(output_dir_wave, exist_ok=True)

                file_prefix = os.path.splitext(filename)[0]

                print(f"Elaborazione: {filename} -> {rel_path}")

                try:
                    create_spectrograms(file_path, output_dir_spec, file_prefix, segment_sec)
                    create_waveforms(file_path, output_dir_wave, file_prefix, segment_sec)
                except Exception as e:
                    print(f"Errore critico su {filename}: {e}")


# ---------------------------------------
# Esecuzione
# ---------------------------------------
if __name__ == "__main__":
    splitted_dataset_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\splitted_dataset"

    spectrogram_output_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Spectrogram"
    waveform_output_dir    = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Waveform"

    if not os.path.exists(splitted_dataset_dir):
        print(f"La cartella '{splitted_dataset_dir}' non esiste!")
    else:
        print("Inizio elaborazione batch (spettrogrammi + waveform in chunk da 5 secondi)...")

        batch_process(
            input_root_folder=os.path.join(splitted_dataset_dir, "train"),
            output_root_folder_spectrogram=os.path.join(spectrogram_output_dir, "train"),
            output_root_folder_waveform=os.path.join(waveform_output_dir, "train"),
            segment_sec=5
        )

        batch_process(
            input_root_folder=os.path.join(splitted_dataset_dir, "test"),
            output_root_folder_spectrogram=os.path.join(spectrogram_output_dir, "test"),
            output_root_folder_waveform=os.path.join(waveform_output_dir, "test"),
            segment_sec=5
        )

        print("Elaborazione completata!")
