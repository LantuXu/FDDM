import os.path
import random

def get_dict_text(text_dir):
    """
    递归读取文件夹及其子文件夹中的所有 .txt 文件，
    提取文件名前缀（去除前导零）和随机一行的内容，存储到字典中。

    :param text_dir: 目标文件夹路径
    :return: 字典，键为文件名前缀，值为随机一行的内容
    """
    dict_text = {}

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(text_dir):
        for file_name in files:
            # 检查文件是否为 .txt 文件
            if file_name.endswith('.txt'):
                # 提取文件名前缀（去除扩展名和前导零）
                base_name = os.path.splitext(file_name)[0].lstrip('0')
                file_path = os.path.join(root, file_name)

                # 读取文件的所有行
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                    # 确保文件不为空
                    if lines:
                        # 生成一个随机行号
                        random_line_number = random.randint(0, len(lines) - 1)
                        # 读取随机行并去除两端空白字符
                        random_line = lines[random_line_number].strip()
                        dict_text[base_name] = random_line

    return dict_text
