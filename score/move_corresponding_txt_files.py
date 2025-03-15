import shutil
import os.path

def move_corresponding_txt_files(a_folder, b_folder, c_folder):
    # 确保目标文件夹存在
    if not os.path.exists(c_folder):
        os.makedirs(c_folder)

    # 获取a文件夹中的所有jpg文件
    jpg_files = [f for f in os.listdir(a_folder) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        # 去掉扩展名得到文件名前缀
        file_prefix = os.path.splitext(jpg_file)[0]
        txt_file = f"{file_prefix}.txt"

        # 遍历b文件夹中的子文件夹，寻找对应的txt文件
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