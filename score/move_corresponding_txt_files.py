import shutil
import os.path

def move_corresponding_txt_files(a_folder, b_folder, c_folder):
    # Make sure the desired folder exists
    if not os.path.exists(c_folder):
        os.makedirs(c_folder)

    # Get all the jpg files in the a folder
    jpg_files = [f for f in os.listdir(a_folder) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        # Remove the extension to get the filename prefix
        file_prefix = os.path.splitext(jpg_file)[0]
        txt_file = f"{file_prefix}.txt"

        # Iterate over the subfolders in the b folder to find the corresponding txt file
        found = False
        for subfolder in os.listdir(b_folder):
            subfolder_path = os.path.join(b_folder, subfolder)
            if os.path.isdir(subfolder_path):
                txt_file_path = os.path.join(subfolder_path, txt_file)
                if os.path.exists(txt_file_path):
                    shutil.copy(txt_file_path, os.path.join(c_folder, txt_file))
                    found = True
                    break

        if not found:
            print(f"{txt_file} does not exist in any subfolder of {b_folder}")