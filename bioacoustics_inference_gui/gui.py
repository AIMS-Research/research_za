import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import threading
import glob, os
from pred_help import *
import tkinter as tk

def browse_folder(textbox):
    current_dir = os.getcwd()
    folder_name = filedialog.askdirectory(initialdir=current_dir+"/",
                                          title="Select a folder")
    if folder_name:
        textbox.config(state=tk.NORMAL)
        textbox.delete("1.0", tk.END)
        textbox.insert(tk.INSERT, folder_name)
        textbox.config(state=tk.DISABLED)

    return folder_name

def browse_file(textbox):
    current_dir = os.getcwd()
    file_path = filedialog.askopenfilename(initialdir=current_dir+"/",
                                           title="Select a file",
                                           filetypes=(("hdf5 files", "*.hdf5"),
                                                      ("All files", "*.*")))
    if file_path:
        textbox.config(state=tk.NORMAL)
        textbox.delete("1.0", tk.END)
        textbox.insert(tk.INSERT, file_path)
        textbox.config(state=tk.DISABLED)

    return file_path

def start_prediction():

    folder_audio_name = textbox_audio.get("1.0", tk.END).strip()
    filename_weights = textbox_model.get("1.0", tk.END).strip()
    folder_output_name = textbox_output.get("1.0", tk.END).strip()



    if not folder_audio_name or not filename_weights or not folder_output_name:
        messagebox.showinfo("Error", "Please select the audio folder, model file, and output folder.")
        return

    def prediction_thread(folder_audio_name, filename_weights, folder_output_name):
        global model

        print('------')
        print(folder_audio_name)
        print(filename_weights)
        print(folder_output_name)
        # Check for missing required inputs
        #if folder_audio_name == "" or filename_weights == "" or folder_output_name == "":
        #    print ('missing')
        #    messagebox.showinfo("showinfo", "Please select an audio folder, a model and an output folder.")
        #    return

        # list the audio files in the folder
        wav_files = glob.glob(folder_audio_name+"/*.wav")
        WAV_files = glob.glob(folder_audio_name+"/*.WAV")
        if wav_files==WAV_files:
            files = wav_files 
        else: 
            files= wav_files+WAV_files

        # Find out how many files in the folder
        number_of_files = len(files)

        # Update the progress bar based on the number of files
        progress_bar['maximum'] = number_of_files

        # Update state of buttons
        button_start["state"] = "disabled"

        # Set the weights name file
        predict.setweights_name(filename_weights)

        # Load the tensorflow model by reading in the weights
        model = predict.load_model()

        # Predict on each file
        for wav in files:

            print('Predicting on file: ', wav)
            predict.process_one_file(wav, model, folder_output_name)

            #f predict.still_running() == False:
            #    predict.reset()
            #    break

            progress_bar['value'] += 1

        print('leaving loop')

        button_start["state"] = "normal"

    # Create and start a new thread for prediction
    thread = threading.Thread(target=prediction_thread, args=(folder_audio_name, filename_weights, folder_output_name))
    thread.start()

def show_about():
    messagebox.showinfo("About", "Lemur Classifier\nVersion 1.0\n\nThis application performs classification on audio files.\n\nMachine Learning for Ecology Group\n\nAIMS South Africa\n\ngithub@aims.ac.za")

def exit_application():
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("Lemur Classifier")

window.configure()

# Set window size and position
window.geometry("550x300")
window.eval('tk::PlaceWindow . center')
window.resizable(False, False)

# Create a menu bar
menubar = tk.Menu(window)
window.config(menu=menubar)

# Create the "File" menu
file_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Exit", command=exit_application)

# Create the "About" menu
about_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="About", menu=about_menu)
about_menu.add_command(label="About", command=show_about)

# Create a frame for the content
content_frame = tk.Frame(window)
content_frame.pack(pady=20)
content_frame.configure()

# Create labels
tk.Label(content_frame, text="Audio Folder:", font=("Helvetica", 12)).grid(row=0, column=0, sticky=tk.W, padx=20, pady=10)
tk.Label(content_frame, text="Model File:",  font=("Helvetica", 12)).grid(row=1, column=0, sticky=tk.W, padx=20, pady=10)
tk.Label(content_frame, text="Output Folder:", font=("Helvetica", 12)).grid(row=2, column=0, sticky=tk.W, padx=20, pady=10)

# Create textboxes
textbox_audio = tk.Text(content_frame, height=1, width=30, font=("Helvetica", 12))
textbox_audio.grid(row=0, column=1, padx=20)
textbox_model = tk.Text(content_frame, height=1, width=30, font=("Helvetica", 12))
textbox_model.grid(row=1, column=1, padx=20)
textbox_output = tk.Text(content_frame, height=1, width=30, font=("Helvetica", 12))
textbox_output.grid(row=2, column=1, padx=20)

# Create browse buttons
button_browse_audio = ttk.Button(content_frame, text="Browse", command=lambda: browse_folder(textbox_audio))
button_browse_audio.grid(row=0, column=2, padx=20)
button_browse_model = ttk.Button(content_frame, text="Browse", command=lambda: browse_file(textbox_model))
button_browse_model.grid(row=1, column=2, padx=20)
button_browse_output = ttk.Button(content_frame, text="Browse", command=lambda: browse_folder(textbox_output))
button_browse_output.grid(row=2, column=2, padx=20)

# Create the start button
button_start = ttk.Button(content_frame, text="Start Prediction", command=start_prediction)
button_start.grid(row=3, column=0, columnspan=3, pady=20)

# Create the progress bar
progress_bar = ttk.Progressbar(window, length=400, mode='determinate')
progress_bar.pack(pady=10)

folder_audio_name = "Please select the audio folder"
filename_weights = "Please select the weights"
folder_output_name = "Please select an output folder"
textbox_audio.insert(tk.INSERT, folder_audio_name)
textbox_model.insert(tk.INSERT, filename_weights)
textbox_output.insert(tk.INSERT, folder_output_name)
textbox_output.update()
textbox_model.update()
textbox_audio.update()

lowpass_cutoff = 4000 # Cutt off for low pass filter (was 4000)
downsample_rate = 9600 # Frequency to downsample to (was 9600)
nyquist_rate = 4800 # Nyquist rate (half of sampling rate) (was 4800)
segment_duration = 4 # how long should a segment be
n_fft = 1024 # Hann window length
hop_length = 256 # Sepctrogram hop size
n_mels = 128 # Spectrogram number of mells
f_min = 500 # Spectrogram, minimum frequency for call
f_max = 9000 # Spectrogram, maximum frequency for call

predict = Prediction(lowpass_cutoff, 
            downsample_rate, nyquist_rate, 
            segment_duration, n_fft, 
            hop_length, n_mels, f_min, f_max)

window.mainloop()
