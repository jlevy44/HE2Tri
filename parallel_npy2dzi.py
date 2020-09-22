import os
import time
import subprocess
from multiprocessing import Process

import fire

import numpy as np
import PIL
from PIL import Image

import os
import sys
import deepzoom
import time
import shutil

def worker(wsi_name, web_dir, shrink_factor):
    # disable safety checks for large images
    PIL.Image.MAX_IMAGE_PIXELS = None

    assert(wsi_name[-4:] == ".npy")

    wsi_prefix = wsi_name[:-4]

    prefix_path = os.path.join(web_dir, "images", wsi_prefix)
    npy_path = prefix_path + ".npy"
    png_path = prefix_path + ".png"
    dzi_path = prefix_path + ".dzi"
    base_html_path = "npy2dzi.html"
    new_html_path = os.path.join(web_dir, "index.html")
    openseadragon_src = "openseadragon/"
    openseadragon_dst = os.path.join(web_dir, "openseadragon/")

    iter_name = web_dir[web_dir.rindex("/") + 1:]
    title = iter_name + " " + wsi_name

    START_TIME = time.time()
    print("Loading .npy file")
    img = np.load(npy_path)
    print("Execution time (s):", time.time() - START_TIME)
    print("Done.\n")

    START_TIME = time.time()
    print("Reducing .npy file")
    if shrink_factor == 1:
        comp_img = img
    else:
        comp_img = np.zeros((img.shape[0] // shrink_factor, img.shape[1] // shrink_factor, img.shape[2]), dtype=np.uint32)
        print(comp_img.shape)
        for i in range(shrink_factor):
            j1 = img.shape[0] - img.shape[0] % shrink_factor
            j2 = img.shape[1] - img.shape[1] % shrink_factor
            comp_img += img[i:j1:shrink_factor, i:j2:shrink_factor]
        comp_img //= shrink_factor
    print("Execution time (s):", time.time() - START_TIME)
    print("Done.\n")

    # create png files
    START_TIME = time.time()
    print("Creating .png file")
    Image.fromarray(comp_img.astype(np.uint8)).save(png_path, compress_level=1)
    print("Execution time (s):", time.time() - START_TIME)
    print("Done.\n")

    # create dzi files
    START_TIME = time.time()
    print("Creating .dzi file")
    creator = deepzoom.ImageCreator(
        tile_size=256,
        tile_overlap=0,
        tile_format="png",
        image_quality=1.0,
    )
    creator.create(png_path, dzi_path)
    print("Execution time (s):", time.time() - START_TIME)
    print("Done.\n")

    START_TIME = time.time()
    print("Creating HTML files")
    # create html files
    with open(base_html_path, "r") as f:
        HTML_STR = "".join(f.readlines())
    HTML_STR = HTML_STR.replace("{REPLACE_wsi_prefix}", os.path.join("images", wsi_prefix))
    HTML_STR = HTML_STR.replace("{REPLACE_title}", title)
    with open(new_html_path, "w") as f:
        f.write(HTML_STR)
    # copy openseadragon
    if not os.path.isdir(openseadragon_dst):
        shutil.copytree(openseadragon_src, openseadragon_dst)
    print("Execution time (s):", time.time() - START_TIME)
    print("Done.\n")


def main(wsi_prefix, model_name, shrink_factor, start_iter, end_iter, incr_iter):
    PROGRAM_START_TIME = time.time()


    START_ITER = 1003200
    END_ITER = 1104000
    INCR_ITER = 4800

    SHRINK_FACTOR = 2
    WSI_NAME = str(wsi_prefix) + "_converted.npy"

    jobs = []
    for i in range(START_ITER, END_ITER + 1, INCR_ITER):
        WEB_DIR = "./results/" + model_name + "/test_latest_iter" + str(i)
        p = Process(target=worker, args=(WSI_NAME, WEB_DIR, shrink_factor))
        jobs += [p]
        p.start()


if __name__=="__main__":
    fire.Fire(main)
