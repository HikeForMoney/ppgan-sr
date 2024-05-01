
# python tools/main.py --config-file configs/realsr_bicubic_noise_x4_df2k.yaml --evaluate-only --load ./weights/esrgan_psnr_x4.pdparams


python -m pudb tools/main.py --config-file configs/esrgan_psnr_x4_div2k.yaml \
                     --evaluate-only --load ./output_dir/esrgan_psnr_x4_div2k-2024-05-01-14-35/iter_1000_weight.pdparams
