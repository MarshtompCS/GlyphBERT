import os
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont, TTCollection

from tqdm import tqdm

img_size = 48
bmp_path = "../data/pinyin_bmp"


def support_this_word(font, word, is_ttf=True):
    if len(word) != 1:
        return False
    unimap = font['cmap'].tables[0].ttFont.getBestCmap() if is_ttf else font[0]['cmap'].tables[0].ttFont.getBestCmap()
    if ord(word) not in unimap.keys():
        return False
    return True


def draw_bmp(img_font, w):
    if len(w) > 1:
        cur_size = int(img_size / (len(w) * 0.5))
    else:
        cur_size = img_size
    font = ImageFont.truetype("./font/simsun.ttc", cur_size)
    im = Image.new("1", (img_size, img_size), 1)
    dr = ImageDraw.Draw(im)
    dr.text((0, 0), w, font=font, fill="#000000", align='center')
    return im


def draw_mask():
    font = ImageFont.truetype("./font/simsun.ttc", int(img_size / 2))
    im = Image.new("1", (img_size, img_size), 1)
    dr = ImageDraw.Draw(im)
    dr.text((0, 8), 'MASK', font=font, fill="#000000", align='center')
    im.save("4.bmp")


def create_special_token_bmp():
    # special_token = ["PAD", "UNK", "CLS", "SEP", "MASK"]
    special_token = ["", "UNK", "CLS", "SEP", ""]
    font = ImageFont.truetype("./font/simsun.ttc", int(img_size / 1.5))
    for idx, text in enumerate(special_token):
        im = Image.new("1", (img_size, img_size), 1)
        dr = ImageDraw.Draw(im)
        dr.text((0, 4), text, font=font, fill="#000000", align='center')
        # im.save(os.path.join(bmp_path, "{}.bmp".format(idx)))
        im.save("{}.bmp".format(idx))


def create_bmp_vocab():
    font_path_list = [
        "simhei.ttf",
        "simsun.ttc",
        "Deng.ttf",
        "msyh.ttc",

        # 符号字体
        # "symbol.ttf",
        # "webdings.ttf",
        # "wingding.ttf"
    ]
    font_path_list = [os.path.join("./font", f) for f in font_path_list]
    font_dict = dict([(f, TTFont(f)) if 'ttf' in f else (f, TTCollection(f)) for f in font_path_list])
    image_font_dict = dict({f: ImageFont.truetype(f, img_size) for f in font_path_list})

    all_vocab = dict()
    with open("../data/pinyin_vocab_bmp.txt", 'r', encoding='utf-8') as reader:
        for idx, i in enumerate(reader.readlines()):
            all_vocab[idx] = i.strip()

    vocab = []

    font_cnt = {f: 0 for f in font_path_list}

    for idx, w in tqdm(all_vocab.items()):
        for f_path, font in font_dict.items():
            if all(support_this_word(font, char, is_ttf=True if 'ttf' in f_path else False) for char in w):
                bmp = draw_bmp(image_font_dict[f_path], w)
                bmp.save(os.path.join(bmp_path, "{}.bmp".format(len(vocab))))
                font_cnt[f_path] += 1
                vocab.append(w)
                break

    with open('../data/pinyin_vocab_bmp.txt', 'w', encoding='utf-8') as writer:
        for i in vocab:
            writer.write(i + '\n')
    print("vocab size: {}/{}".format(len(vocab), len(all_vocab)))
    print("font used counter:")
    for item in font_cnt.items():
        print(item)


if __name__ == '__main__':
    create_bmp_vocab()
    # create_special_token_bmp()
    # draw_mask()
