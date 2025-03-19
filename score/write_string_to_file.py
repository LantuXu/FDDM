import os.path

def write_string_to_file(folder_path, file_name, content):
    # Concatenate file paths
    file_path = os.path.join(folder_path, file_name)

    # Create a folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Writing to a file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")