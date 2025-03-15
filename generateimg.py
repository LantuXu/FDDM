from generate.gen_class import *
from omegaconf import OmegaConf
from score import *

if __name__ == '__main__':
    for j in range(0, 5):
        print(f"times: {j}")
        with open("score.txt", "a") as file:
            file.write(f"times: {j}\n")

        config_path = "CLIP_hyper.yaml"
        config = OmegaConf.load(config_path)  # 加载配置文件
        model = gen_class(**config.get("model").get("params", dict()))

        model.train_step()

        b_size = 20  # 每批生成多少图片
        b_num = 50000 // b_size  # 每次生成多少批图片(向下取整)
        b_rem = 50000 % b_size  # 批次余数

        gen_txts = random_select_lines(model.gen_txt_path, b_num)  # 选取随机n行描述
        i = 0

        fake_directory = f"/root/autodl-tmp/score_img{j}"       # 替换为你需要的图像生成路径
        txt_path_fake = f"/root/autodl-tmp/score_txt{j}"        # 替换为你的文字保存路径

        for gen_txt in gen_txts:
            model.predict(gen_txt, fake_directory, b_size, x=i)  # 生成n张图片到fakepath
            for j in range(b_size):
                write_string_to_file(txt_path_fake, f"output_image_{i}_{j}.txt", gen_txt)  # 将对应的描述写入faketxt
            i = i + 1

        if b_rem > 0:
            model.predict(gen_txts[0], fake_directory, b_rem, x=i)  # 生成n张图片到fakepath
            for j in range(b_rem):
                write_string_to_file(txt_path_fake, f"output_image_{i}_{j}.txt", gen_txts[0])  # 将对应的描述写入faketxt

        print("done")