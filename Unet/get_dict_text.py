import os.path
import random

def get_dict_text(text_dir):
    dict_text = {}
    for root, dirs, files in os.walk(text_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                base_name = os.path.splitext(file_name)[0].lstrip('0')
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    if lines:
                        # Generate a random line number
                        random_line_number = random.randint(0, len(lines) - 1)
                        # Read a random line and remove whitespace from both ends
                        random_line = lines[random_line_number].strip()
                        dict_text[base_name] = random_line

    return dict_text
