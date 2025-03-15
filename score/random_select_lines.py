import random

def random_select_lines(file_path, n):
    """
    从指定的 .txt 文件中随机选取 n 行，并存储在数组中。

    :param file_path: .txt 文件的路径
    :param n: 需要随机选取的行数
    :return: 包含随机选取的 n 行的数组
    """
    selected_lines = []  # 存储随机选取的行

    try:
        # 打开文件并读取所有行
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 检查文件是否为空
        if not lines:
            print("The file is empty.")
            return selected_lines

        # 检查 n 是否大于文件的总行数
        if n > len(lines):
            print(f"Warning: The file only has {len(lines)} lines. Selecting all lines.")
            n = len(lines)

        # 随机选取 n 行
        selected_lines = random.sample(lines, n)

        # 去除每行的换行符
        selected_lines = [line.strip() for line in selected_lines]

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return selected_lines