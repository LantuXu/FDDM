import os

def get_d1ct_text(text_dir):
    dict_text = {}
    # 遍历指定目录及其子目录
    for class_dir in os.listdir(text_dir):
        class_path = os.path.join(text_dir, class_dir)

        # 检查是否为目录
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                base_name = file_name.split('.')[0].lstrip('0')
                file_path = os.path.join(class_path, file_name)
                # 检查是否为文件
                if os.path.isfile(file_path) and file_name.endswith('.txt'):
                    # 读取文件的第一行
                    with open(file_path, 'r', encoding='utf-8') as file:
                        first_line = file.readline().strip()
                        dict_text[base_name] = first_line
        else: # 是文件
            base_name = class_dir.split('.')[0].lstrip('0')
            file_path = os.path.join(class_path, base_name)
            if os.path.isfile(class_path) and class_dir.endswith('.txt'):
                # 读取文件的第一行
                with open(class_path, 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip()
                    dict_text[base_name] = first_line
    return dict_text