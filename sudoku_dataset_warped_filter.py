import cv2
import re
import os
import shutil
from pynput import keyboard
import threading
from python_helper_functions import detectSudokuPuzzle

last_key_pressed = None

def on_press(key):
    global last_key_pressed
    last_key_pressed = key.char
    print('Key {} pressed'.format(key))
    # Stop listener after the first key press
    return False

def key_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    
    original_path = "./original_dataset"
    pattern = r"\.jpg$"

    original_filenames = os.listdir(original_path)
    original_filenames = [filename for filename in original_filenames if re.search(pattern, filename)]

    print(f"Number of training images {len(original_filenames)}")

    for i in range(len(original_filenames)):

        curr_img_path = "./original_dataset/" + original_filenames[i]
        curr_dat_path = "./original_dataset/" + original_filenames[i].replace(".jpg", ".dat")
        new_img_path = "./warped_dataset/" + original_filenames[i]
        new_dat_path = "./warped_dataset/" + original_filenames[i].replace(".jpg", ".dat")
        print(f"{i} | {curr_img_path}")
        result = detectSudokuPuzzle(curr_img_path, True)

        while True:

            # Create and start the thread for key listener
            listener_thread = threading.Thread(target=key_listener)
            listener_thread.start()

            # Wait for the thread to finish
            listener_thread.join()

            if last_key_pressed == 'y':
                print(f"{curr_img_path} was selected")
                shutil.copyfile(curr_dat_path, new_dat_path)
                cv2.imwrite(new_img_path, result)
                break
            elif last_key_pressed == 'n':
                print(f"{curr_img_path} was NOT selected")
                break
            else:
                print(f"Please enter y/n")