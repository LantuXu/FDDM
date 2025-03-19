from generate.gen_class import *
from omegaconf import OmegaConf
from score import *

if __name__ == '__main__':
    for j in range(0, 5):
        print(f"times: {j}")
        with open("score.txt", "a") as file:
            file.write(f"times: {j}\n")

        config_path = "CLIP_hyper.yaml"
        config = OmegaConf.load(config_path)  # Loading configuration files
        model = gen_class(**config.get("model").get("params", dict()))

        model.train_step()

        b_size = 20  # Number of images generated per batch
        b_num = 50000 // b_size  # Number of batches per generation (rounded down)
        b_rem = 50000 % b_size  # Batch remainder

        gen_txts = random_select_lines(model.gen_txt_path, b_num)  # Random n lines of description are selected
        i = 0

        fake_directory = f"/root/autodl-tmp/score_img{j}"       # paths for the Generated images
        txt_path_fake = f"/root/autodl-tmp/score_txt{j}"        # Path for saved the texts

        for gen_txt in gen_txts:
            model.predict(gen_txt, fake_directory, b_size, x=i)  # Generate n images to fakepath
            for j in range(b_size):
                write_string_to_file(txt_path_fake, f"output_image_{i}_{j}.txt", gen_txt)  # Write the corresponding description to faketxt
            i = i + 1

        if b_rem > 0:
            model.predict(gen_txts[0], fake_directory, b_rem, x=i)  # Generate n images to fakepath
            for j in range(b_rem):
                write_string_to_file(txt_path_fake, f"output_image_{i}_{j}.txt", gen_txts[0])  # Write the corresponding description to faketxt

        print("done")