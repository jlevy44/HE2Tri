MODEL_NAME=he2tri
MODEL_PATH=/path/to/model.pth
WSI_PATH=/path/to/npy.npy
WINDOW_SIZE=256
STEP_SIZE=256

nohup python test_wsi.py --model_loc G_A:$MODEL_PATH --dataroot ./datasets/test_images/ --dataset_mode npy --name $MODEL_NAME --phase test --wsi_name $NPY_PATH --no_dropout --model_suffix G,G_A --iter_start 0 --load_iter 0 --iter_incr 0 --load_size $WINDOW_SIZE --crop_size $WINDOW_SIZE --dps $STEP_SIZE --shrink_factor 1 --model test --gpu_ids 0 &
