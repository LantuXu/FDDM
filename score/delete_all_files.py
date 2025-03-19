import shutil
import os.path

def delete_all_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return

    # Iterate over all files and subfolders in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Delete it if it's a file
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If it's a folder, delete its contents recursively
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"An error occurred while deleting {file_path}: {e}")