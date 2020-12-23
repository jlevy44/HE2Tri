# HE2Tri

Article: https://www.nature.com/articles/s41379-020-00718-1

Credits:  
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
- Richard Zhan  

Steps:  
1. Train model and export WSI realtime with train.py (uncomment and manipulate code in train.py to build viewable WSI per model iteration during training)
2. Convert entire WSI realtime using test_wsi.py  
3. Convert translated NPY into DZI with npy2dzi.py, parallel_npy2dzi.py or npy2dzi_orig.py.  
4. Convert translated NPY into TIFF with npy2tif.py.
5. View TIFF files with ASAP, or DZI files with OpenSeaDragon.
6. Split view H&E and vTrichrome side-by-side with openseadragon curtain functionality by modifying index.html.   
