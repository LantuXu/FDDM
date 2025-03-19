import random

def random_select_lines(file_path, n):
    selected_lines = []  # Stores randomly selected rows

    try:
        # Open the file and read all the lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if not lines:
            print("The file is empty.")
            return selected_lines

        # Check that n is greater than the total number of lines in the file
        if n > len(lines):
            print(f"Warning: The file only has {len(lines)} lines. Selecting all lines.")
            n = len(lines)

        # Select n rows at random
        selected_lines = random.sample(lines, n)

        # Remove newline characters from each line
        selected_lines = [line.strip() for line in selected_lines]

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return selected_lines