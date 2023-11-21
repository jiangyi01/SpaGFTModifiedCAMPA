from campa.tl import LossEnum, ModelEnum

base_config = {
    "experiment": {
        "dir": "test",
        "name": None,
        "save_config": True,
    },
    "data": {
        "data_config": "ExampleData",
        "dataset_name": "184A1_test_dataset",
        "output_channels": None,
    },
    "model": {
        "model_cls": None,
        "model_kwargs": {
            "num_neighbors": 3,
            "num_channels": 34,
            "num_output_channels": 34,
            "latent_dim": 16,
            # encoder definition
            "encoder_conv_layers": [32],
            "encoder_conv_kernel_size": [1],
            "encoder_fc_layers": [32, 16],
            # decoder definition
            "decoder_fc_layers": [],
        },
        # if true, looks for saved weights in experiment_dir
        # if a path, loads these weights
        "init_with_weights": False,
    },
    "training": {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 128,
        "loss": {"decoder": LossEnum.SIGMA_MSE, "latent": LossEnum.KL},
        "metrics": {"decoder": LossEnum.MSE_metric, "latent": LossEnum.KL},
        # saving models
        "save_model_weights": True,
        "save_history": True,
        "overwrite_history": True,
    },
    "evaluation": {
        "split": "val",
        "predict_reps": ["latent", "decoder"],
        "img_ids": 2,
        "predict_imgs": True,
        "predict_cluster_imgs": True,
    },
    "cluster": {  # cluster config, also used in this format for whole data clustering
        "cluster_name": "clustering",
        "cluster_rep": "latent",
        "cluster_method": "leiden",  # leiden or kmeans
        "leiden_resolution": 0.2,
        "subsample": None,  # 'subsample' or 'som'
        "subsample_kwargs": {},
        "som_kwargs": {},
        "umap": True,
    },
}


variable_config = [
    # conditional VAE model
    {
        "experiment": {"name": "CondVAE_pert-CC-Initial"},
        "model": {
            "model_cls": ModelEnum.VAEModel,
            "model_kwargs": {
                "num_conditions": 14,
                "encode_condition": [10, 10],
            },
        },
    },
    # conditional VAE model with SpaGFT
    {
        "experiment": {"name": "CondVAE_pert-CC-SpaGFT-Modified"},
        "model": {
            "model_cls": ModelEnum.VAEModel,
            "model_kwargs": {
                "num_conditions": 14,
                "encode_condition": [10, 10],
            },
        },
    },

    
    

]
