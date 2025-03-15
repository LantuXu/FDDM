import os.path

def write_string_to_file(folder_path, file_name, content):
    """
    将字符串内容写入指定文件夹下的指定文件中。如果文件不存在，则创建文件。

    :param folder_path: 文件夹路径
    :param file_name: 文件名
    :param content: 要写入的字符串内容
    """
    # 拼接文件路径
    file_path = os.path.join(folder_path, file_name)

    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)

    # 写入文件
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")