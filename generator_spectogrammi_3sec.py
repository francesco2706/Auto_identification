import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def get_3sec_segments(y, sr, segment_sec=3):
    segment_samples = sr * segment_sec 
    segments = []
    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        if end <= len(y):
            segments.append(y[start:end])
    return segments

def create_spectrograms(audio_path, output_dir, file_prefix, segment_sec=3):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Not possible to load {audio_path}: {e}")
        return
    
    segments = get_3sec_segments(y, sr, segment_sec)

    if len(segments) == 0:
        print(f"Audio too short {audio_path}, skipped.")
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

def batch_process(input_root_folder, output_root_folder_spectrogram, segment_sec=3):
    valid_extensions = ('.wav')

    for root, dirs, files in os.walk(input_root_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):

                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_root_folder)

                output_dir_spec = os.path.join(output_root_folder_spectrogram, rel_path)
                os.makedirs(output_dir_spec, exist_ok=True)

                file_prefix = os.path.splitext(filename)[0]

                print(f"Processing: {filename} -> {rel_path}")

                try:
                    create_spectrograms(file_path, output_dir_spec, file_prefix, segment_sec)
                except Exception as e:
                    print(f"Error on {filename}: {e}")

if __name__ == "__main__":
    splitted_dataset_dir = r"C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto finale\Data\dataset_da_splittare"
    base_output_dir = r"C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto finale"
    new_folder_name = "Dataset_Spectrogram_3sec"
    final_output_root = os.path.join(base_output_dir, new_folder_name)

    if not os.path.exists(splitted_dataset_dir):
        print(f"The folder '{splitted_dataset_dir}' does not exist!")
    else:
        print(f"Start processing... Output will be in: {final_output_root}")
        batch_process(
            input_root_folder=os.path.join(splitted_dataset_dir, "train"),
            output_root_folder_spectrogram=os.path.join(final_output_root, "train"),
            segment_sec=3
        )

        batch_process(
            input_root_folder=os.path.join(splitted_dataset_dir, "test"),
            output_root_folder_spectrogram=os.path.join(final_output_root, "test"),
            segment_sec=3
        )
        
        print("Done")

