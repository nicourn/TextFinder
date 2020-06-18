import cv2
import numpy as np
import pytesseract
import os
import time

class Player:
    def __init__(self, name, bank):
        self.name = str(int(name) + 1)
        self.bank = bank

    def __str__(self):
        return f"{self.name}: {self.bank}"

def get_text(image, shadow):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = image.shape[:2]
    y, x = size[0] * 3, size[1] * 4
    image = cv2.resize(image, (x, y))
    if shadow:
        _, image = cv2.threshold(image, 75, 255, 0)
    else:
        _, image = cv2.threshold(image, 137, 255, 0)
    # cv2.imshow("Bank", image)
    # cv2.waitKey()
    text = pytesseract.image_to_string(image)
    return text

def for_find_copy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 50, 255, 0)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


bases = [cv2.imread("base.png"), cv2.imread("basel.png"),
         cv2.imread("lbase.png"), cv2.imread("lbasel.png")]

fish = cv2.imread("fish.png")
fish = cv2.cvtColor(fish, cv2.COLOR_BGR2GRAY)
fish = cv2.GaussianBlur(fish, (3, 3), 0)

players = []

for i, base in enumerate(bases):
    bases[i] = for_find_copy(base)


h, w = bases[0].shape[:2]

os.chdir("test")
for i in os.listdir():
    start = time.time()
    image = cv2.imread(i)
    rectsw = []
    rectsh = []
    g_img = for_find_copy(image)
    fish_i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fish_i = cv2.GaussianBlur(fish_i, (3, 3), 0)


    for n, pbase in enumerate(bases):
        t = 3 if n < 2 else 1
        for i in range(t):
            find = cv2.matchTemplate(g_img, pbase, cv2.TM_SQDIFF)
            score, _, minc, _ = cv2.minMaxLoc(find)
            cv2.rectangle(g_img, minc, (minc[0] + w, minc[1] + h), (0, 0, 255), -1)
            if minc[0] in rectsw and minc[1] in rectsh:
                continue
            else:
                rectsh += [h for h in range(minc[1] + h * (-1), minc[1] + h * 1)]
                rectsw += [w for w in range(minc[0] + w * (-1), minc[0] + w * 1)]
                cv2.rectangle(image, (minc[1] + h * (-1), minc[1] + h * 1),(minc[0] + w * (-1), minc[0] + w * 1), (0, 0, 255), -1)
                pass

            cv2.rectangle(image, minc, (minc[0] + w, minc[1] + h), (0, 0, 255), 1)
            if n == 0:
                bank_image = image[minc[1] + (h//2): minc[1] + h, minc[0] - (w * 3): minc[0] + (w // 5)]
            else:
                bank_image = image[minc[1] + (h//2): minc[1] + h , minc[0] + w: minc[0] + (w * 4)]
            name = str(len(players))
            coord = minc
            if (n + 1) % 2 != 0:
                bank = get_text(bank_image, False)
            else:
                bank = get_text(bank_image, True)
            # print(score / 100000, bank)
            cv2.imwrite(f"{len(os.listdir())}.png", bank_image)
            players.append(Player(name, bank))

    find = cv2.matchTemplate(fish_i, fish, cv2.TM_SQDIFF)
    score, _, minc, _ = cv2.minMaxLoc(find)
    cv2.rectangle(image, minc, (minc[0] + w, minc[1] + h), (0, 0, 255), 1)

    h1, w1 = image.shape[:2]
    lines = get_text(image, False).split('\n')
    lines += get_text(image, True).split('\n')
    a = [text for text in lines if "pot:" in text.lower()]
    print(a)
    print([str(player) for player in players])
    print(f"Time: {time.time() - start}")
    print('*' * 50)
    cv2.imshow("Image", image)
    cv2.waitKey()
    players = []

