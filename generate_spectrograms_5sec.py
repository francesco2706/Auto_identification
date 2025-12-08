
#Pre-processing of our audio dataset. Main function: transofrm the audio in spectrograms, 
#by segmenting the audio into fixed-length (5 seconds). We use librosa library, that is 
#the specialized library for audio and music analysis, used for loading audio,
#calculating the Short-Time Fourier Transform and visualization. 
import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#Segmenting the audio: y = audio time series data; sr = sample rate
def get_5sec_segments(y, sr, segment_sec=5):
    segment_samples = sr * segment_sec #calculating the required number of samples for a 5-second segment
    segments = []
    #iterate through the audio array, slicing out segments. 
    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        #only segments with exact full length are included, discarding any shorter segment at the end of the file
        if end <= len(y):
            segments.append(y[start:end])
    return segments


#This function read an audio file and convert each 5-second segment into a spectogram image
def create_spectrograms(audio_path, output_dir, file_prefix, segment_sec=5):
    try:
        y, sr = librosa.load(audio_path, sr=None) #loading audio
    except Exception as e:
        print(f"Not possible to load {audio_path}: {e}")
        return
    #break the audio
    segments = get_5sec_segments(y, sr, segment_sec)

    if len(segments) == 0:
        print(f"Audio too short {audio_path}, skipped.")
        return

    #for each segment 
    for i, y_segment in enumerate(segments):
        #calculate the Short-Time Fourier Transform, to convert the signal from the time
        #domain to the frequency domain 
        D = librosa.stft(y_segment)
        #converts the magnitude of the STFT result into the decible dB scale
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        #plotting and saving 
        fig, ax = plt.subplots(figsize=(4,4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)

        save_name = f"{file_prefix}_part{i+1}.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)

# Function to manage the entire process
def batch_process(input_root_folder, output_root_folder_spectrogram, segment_sec=5):
    valid_extensions = ('.wav')

    for root, dirs, files in os.walk(input_root_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):

                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_root_folder)

                # Creating output folders
                output_dir_spec = os.path.join(output_root_folder_spectrogram, rel_path)
                os.makedirs(output_dir_spec, exist_ok=True)

                file_prefix = os.path.splitext(filename)[0]

                print(f"Processing: {filename} -> {rel_path}")

                try:
                    create_spectrograms(file_path, output_dir_spec, file_prefix, segment_sec)
                except Exception as e:
                    print(f"Error on {filename}: {e}")

if __name__ == "__main__":
    splitted_dataset_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\splitted_dataset"
    spectrogram_output_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\only_Spectrogram"

    if not os.path.exists(splitted_dataset_dir):
        print(f"La cartella '{splitted_dataset_dir}' non esiste!")
    else:
        print("Start processing...")

        batch_process(
            input_root_folder=os.path.join(splitted_dataset_dir, "train"),
            output_root_folder_spectrogram=os.path.join(spectrogram_output_dir, "train"),
            segment_sec=5
        )

        batch_process(
            input_root_folder=os.path.join(splitted_dataset_dir, "test"),
            output_root_folder_spectrogram=os.path.join(spectrogram_output_dir, "test"),
            segment_sec=5
        )

        print("Elaborazione completata!")
