import json
import os

def gen_config(out_path):
    config = dict(
        model_name = "LightFormerPredictor",
        image_num = 10,
        embed_dim = 256,
        num_heads = 8,
        num_sam_pts = 8,
        num_levels = 1,
        num_query = 1,
        log_every_n_steps = 10,
        mlp_out_channel = 1024,
        n = 8,
        out_class_num = 4,
        training = dict(
            sample_database_folder = [
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip1/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip2/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip3/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip4/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip5/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip6/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip7/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip8/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip9/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip10/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip11/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip12/train",
                "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip13/train",
            ],
            batch_size = 8,
            loader_worker_num = 20,
            epoch = 10,
            accelerator = "mps" # https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html
        ),
        validation = dict(
            sample_database_folder = [
                "/Users/gordonliu/Documents/ml_projects/LightForker/dataset/Kaggle_Dataset/daySequence1",
            ],
            batch_size = 8,
            loader_worker_num = 20,
            check_interval = 1.0,
            limit_batches = 1.0
        ),
        test = dict(
            sample_database_folder = [
                "/Users/gordonliu/Documents/ml_projects/LightForker/dataset/Kaggle_Dataset/daySequence2",
            ],
            batch_size = 1,
            loader_worker_num = 20,
            visualization = False,
            test_result_pkl_dir = "/workspace/debug/prediction_ml_framework/pred_res",
        ),
        optim = dict(
            init_lr = 0.0001,
            step_size = 3,
            step_factor = 0.5,
            gradient_clip_val = None,
            gradient_clip_algorithm = "norm"
        )
    )
    with open(os.path.join(out_path, "Light_Former_config.json"), "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    out_path = "/Users/gordonliu/Documents/ml_projects/LightForker/configs"
    gen_config(out_path)
