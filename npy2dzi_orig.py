import glob,os
import numpy as np
import tifffile
import subprocess
import fire
import cv2

REPLACE_STR="""<html>
<div id="openseadragon1" style="width: 100%; height: 100%";></div>
<script src="/openseadragon/openseadragon.js"></script>
<script type="text/javascript">
    var viewer = OpenSeadragon({
        id: "openseadragon1",
        prefixUrl: "/openseadragon/images/",
        tileSources: "/REPLACE.dzi",
        sequenceMode: false,
        autoHideControls: true,
        animationTime: 1.0,
        blendTime: 0.6,
        constrainDuringPan: true,
        maxZoomPixelRatio: 1,
        visibilityRatio: 1,
        zoomPerScrolli: 1,
        defaultZoomLevel: 1,
        showReferenceStrip: true,
        showNavigator:  true,
	    showFullPageControl: false
    });
</script>
</html>
"""
def main(npy="",out_dir='',compression_factor=1.):#/media/joshualevy/Elements1/final_vTrichrome_cGAN/
    print(npy)
    os.makedirs(out_dir,exist_ok=True)
    tif_dir=os.path.join(out_dir,"tiff_files")
    os.makedirs(tif_dir,exist_ok=True)
    out_file=os.path.join(tif_dir,os.path.basename(npy.replace('.npy','.tif').replace('.png','.tif')))
    out_dzi=os.path.join(out_dir,".",os.path.basename(out_file).replace('.tif',''))#dzi_files
    os.makedirs(os.path.dirname(out_dzi),exist_ok=True)
    out_html=os.path.join(out_dir,os.path.basename(out_file).replace('.tif','.html'))
    if not os.path.exists(out_file):
        img=(np.load(npy) if npy.endswith('.npy') else cv2.cvtColor(cv2.imread(npy),cv2.COLOR_BGR2RGB))
        if compression_factor>1.:
            img=cv2.resize(img,None,fx=1./compression_factor,fy=1./compression_factor,interpolation=cv2.INTER_CUBIC)
        tifffile.imwrite(out_file,img, photometric='rgb')
    if not os.path.exists(out_file+'f') and not os.path.exists(out_dzi+'.dzi'):
        subprocess.call("vips tiffsave --compression=lzw --Q=100 --tile --tile-width=512 --tile-height=512 --pyramid --vips-progress {0} {1}".format(out_file,out_file+'f'),shell=True)
    if not os.path.exists(out_dzi+'.dzi'):
        subprocess.call("vips dzsave --tile-size 512 --vips-progress {0} {1} ".format(out_file+'f',out_dzi),shell=True)
    if not os.path.exists(out_html):
        html_text=REPLACE_STR.replace("REPLACE",os.path.basename(out_dzi))
        with open(out_html,'w') as f:
            f.write(html_text)

if __name__=="__main__":
    fire.Fire(main)
