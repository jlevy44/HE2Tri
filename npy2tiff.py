import tifffile
import cv2, numpy as np  # OpenCV for fast resizing
import fire
import os, subprocess

def write_read_pyramid(npy_file, tif_out_file):
    image = np.load(npy_file)#tifffile.imread(tiff_file, key=0)
    h, w, s = image.shape
    new_img_name=tif_out_file#tiff_file+'.tmp.tiff'
    if not os.path.exists(new_img_name):
        with tifffile.TiffWriter(new_img_name, bigtiff=True) as tif:
            level = 0
            while True:
                tif.save(
                    image,
                    software='Glencoe/Faas pyramid',
                    metadata=None,
                    tile=(256, 256),
                    resolution=(1000/2**level, 1000/2**level, 'CENTIMETER'),
                    # compress=1,  # low level deflate
                    # compress=('jpeg', 95),  # requires imagecodecs
                    # subfiletype=1 if level else 0,
                )
                if max(w, h) < 256:
                    break
                level += 1
                w //= 2
                h //= 2
                image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    # return tifffile.TiffFile(new_img_name)

def npy2tif(npy_file, tif_out_file, compression_factor=6., internal_pyramid=False):
    if internal_pyramid:
        write_read_pyramid(npy_file, tif_out_file)
    else:
        if not os.path.exists(tif_out_file+'.tmp'):
            img=(np.load(npy_file) if npy_file.endswith('.npy') else cv2.cvtColor(cv2.imread(npy_file),cv2.COLOR_BGR2RGB))
            if compression_factor>1.:
                img=cv2.resize(img,None,fx=1./compression_factor,fy=1./compression_factor,interpolation=cv2.INTER_CUBIC)
            tifffile.imwrite(tif_out_file+".tmp",img, photometric='rgb')
        if not os.path.exists(tif_out_file):
            subprocess.call("vips tiffsave --compression=lzw --Q=100 --tile --tile-width=512 --tile-height=512 --pyramid --vips-progress {0} {1}".format(tif_out_file+".tmp",tif_out_file),shell=True)

if __name__=="__main__":
    fire.Fire(npy2tif)
