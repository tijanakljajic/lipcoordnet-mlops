gpu = "0"
random_seed = 0
data_type = "coords"
video_path = "demo_data/lip_images/"
val_list = "demo_data/demo_val.txt"
anno_path = "demo_data/GRID_alignments"
coords_path = "demo_data/lip_coordinates"

vid_padding = 75
txt_padding = 200

batch_size = 8  #prethodno 40
base_lr = 2e-5
num_workers = 0 #prehodno 16
max_epoch = 10 #prethodno 10000
display = 25 #prethodno 50
test_step = 750 #prethodno 1000
save_prefix = "../results/egcllc/main_pretrained_baseline/LipCoordNet_coords"
is_optimize = True #prethodno True
pin_memory = True

#weights = "pretrain/LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt"
