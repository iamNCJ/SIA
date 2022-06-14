import json
import os
import shutil

from tqdm import tqdm


def mkdir(path):
    print(f'mkdir {path}')
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'done')


def mv(src, dst):
    shutil.move(src, dst)


if __name__ == '__main__':
    class_index = json.load(open('./class_index.json'))
    class_mapping = {}
    for k, v in class_index.values():
        class_mapping[k] = v
    print(class_mapping)

    mkdir('./data')

    for filename in tqdm(os.listdir('./val_images')):
        # print(filename[24:-5])
        label = class_mapping[filename[24:-5]]
        mkdir(f'./data/{label}')
        mv(f'./val_images/{filename}', f'./data/{label}/{filename}')
