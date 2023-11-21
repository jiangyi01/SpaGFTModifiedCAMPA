from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from campa.tl import Experiment
    import numpy as np
import numpy as np
import os
import logging


import pandas as pd
import tensorflow as tf
import random



from campa.tl import LossEnum, ModelEnum
from campa.data import NNDataset
from campa.tl._layers import UpdateSparsityLevel


# --- Callbacks ---
class LossWarmup(tf.keras.callbacks.Callback):
    """Callback to warmup loss weights."""

    def __init__(self, weight_vars, to_weights, to_epochs):
        super().__init__()
        self.to_weights = to_weights
        self.to_epochs = to_epochs
        self.weight_vars = weight_vars

    def on_epoch_begin(self, epoch, logs=None):
        """Update loss weights."""
        for key in self.to_epochs.keys():
            to_epoch = self.to_epochs[key]
            to_weight = self.to_weights[key]
            if to_epoch == 0 or to_epoch <= epoch:
                tf.keras.backend.set_value(self.weight_vars[key], to_weight)
            else:
                tf.keras.backend.set_value(self.weight_vars[key], to_weight / to_epoch * epoch)
            print(f"set {key} loss weight to {tf.keras.backend.get_value(self.weight_vars[key])}")

        if "latent" in self.weight_vars.keys():
            print(f"set latent loss weight to {tf.keras.backend.get_value(self.weight_vars['latent'])}")


class AnnealTemperature(tf.keras.callbacks.Callback):
    """Callback to anneal learning rate."""

    def __init__(self, temperature, initial_temperature, final_temperature, to_epoch):
        super().__init__()
        self.temperature = temperature
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.to_epoch = to_epoch

    def on_epoch_begin(self, epoch, logs=None):
        """Update temperature."""
        if self.to_epoch == 0 or self.to_epoch <= epoch:
            tf.keras.backend.set_value(self.temperature, self.final_temperature)
        else:
            tf.keras.backend.set_value(
                self.temperature,
                self.initial_temperature + (self.final_temperature - self.initial_temperature) / self.to_epoch * epoch,
            )
        print(f"set temperature to {tf.keras.backend.get_value(self.temperature)}")


# --- Estimator class ---
class Estimator_spagft:
    """
    Neural network estimator.

    Handles training and evaluation of models.

    Parameters
    ----------
    exp
        Experiment with model config.
    """

    def __init__(self, exp: Experiment):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp
        print(self.exp)
        self.config = exp.estimator_config
        

        self.config["training"]["loss"] = {
            key: LossEnum(val).get_fn() for key, val in self.config["training"]["loss"].items()
        }
        self.config["training"]["metrics"] = {
            key: LossEnum(val).get_fn() for key, val in self.config["training"]["metrics"].items()
        }
        self.callbacks: list[object] = []

        # create model
        self.optimizer = None
        self.epoch = 0
        self.create_model()
        self.compiled_model = False

        # train and val datasets
        # config params impacting y
        self.output_channels = self.config["data"]["output_channels"]
        self.repeat_y = len(self.config["training"]["loss"].keys())
        if self.repeat_y == 1:
            self.repeat_y = False
        self.add_c_to_y = False
        if "adv_head" in self.config["training"]["loss"].keys():
            self.add_c_to_y = True
            self.repeat_y = self.repeat_y - 1
        self.ds = NNDataset(
            self.config["data"]["dataset_name"],
            data_config=self.config["data"]["data_config"],
        )
        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None

        # set up model weights and history paths for saving/loading later
        self.weights_name = os.path.join(self.exp.full_path, "weights_epoch{:03d}")  # noqa: P103
        self.history_name = os.path.join(self.exp.full_path, "history.csv")

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """
        Shuffled :class:`tf.data.Dataset` of train split.
        """
        if self._train_dataset is None:
            self._train_dataset = self._get_dataset("train", shuffled=True)
        return self._train_dataset

    @property
    def val_dataset(self) -> tf.data.Dataset:
        """
        :class:`tf.data.Dataset` of val split.
        """
        if self._val_dataset is None:
            self._val_dataset = self._get_dataset("val")
        return self._val_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        """
        :class:`tf.data.Dataset` of test split.
        """
        if self._test_dataset is None:
            self._test_dataset = self._get_dataset("test")
        return self._test_dataset

    def _get_dataset(self, split: str, shuffled: bool = False) -> tf.data.Dataset:
        return self.ds.get_tf_dataset(
            split=split,
            output_channels=self.output_channels,
            is_conditional=self.model.is_conditional,
            repeat_y=self.repeat_y,
            add_c_to_y=self.add_c_to_y,
            shuffled=shuffled,
        )

    def create_model(self):
        """
        Initialise neural network model.

        Adds ``self.model``.
        """
        ModelClass = ModelEnum(self.config["model"]["model_cls"]).get_cls()
        self.model = ModelClass(**self.config["model"]["model_kwargs"])
        weights_path = self.config["model"]["init_with_weights"]
        if weights_path is True:
            weights_path = tf.train.latest_checkpoint(self.exp.full_path)
            if weights_path is None:
                self.log.warning(
                    f"WARNING: weights_path set to true but no trained model found in {self.exp.full_path}"
                )
        if isinstance(weights_path, str):
            # first need to compile the model
            self._compile_model()
            self.log.info(f"Initializing model with weights from {weights_path}")
            w1 = self.model.model.layers[5].get_weights()
            self.model.model.load_weights(weights_path).assert_nontrivial_match().assert_existing_objects_matched()
            w2 = self.model.model.layers[5].get_weights()
            assert (w1[0] != w2[0]).any()
            assert (w1[1] != w2[1]).any()
            self.epoch = self.exp.epoch
            # TODO when fine-tuning need to reset self.epoch!

    def _compile_model(self):
        config = self.config["training"]
        # set loss weights
        self.loss_weights = {key: tf.keras.backend.variable(val) for key, val in config["loss_weights"].items()}
        # callback to update weights before each epoch
        self.callbacks.append(
            LossWarmup(
                self.loss_weights,
                config["loss_weights"],
                config["loss_warmup_to_epoch"],
            )
        )
        self.callbacks.append(UpdateSparsityLevel())
        if hasattr(self.model, "temperature"):
            self.callbacks.append(
                AnnealTemperature(
                    self.model.temperature,
                    self.model.config["initial_temperature"],
                    self.model.config["temperature"],
                    self.model.config["anneal_epochs"],
                )
            )
        # create optimizer
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        self.model.model.compile(
            optimizer=self.optimizer,
            loss=config["loss"],
            loss_weights=self.loss_weights,
            metrics=config["metrics"],
        )
        self.compiled_model = True
    def random_seed(self,seed):
        os.environ["PYTHONHASHSEED"] = "42"
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42) 
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    def train_model_with_spagft(self,lamda=0.5,use_stored_data=False, stored_data_path=None, gft_script_path=None,gft_env_path=None):
        self.random_seed(42)
        def gaussian_nll(mu, log_sigma, x):
            """Gaussian negative log-likelihood.
            From: https://github.com/orybkin/sigma-vae-tensorflow/blob/master/model.py
            """
            return 0.5 * ((x - mu) / tf.math.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)
        @tf.function
        def min_entropy(y_true, y_pred):
            """
            Entropy
            """
            l_ent = -1 * tf.reduce_mean(tf.math.log(tf.nn.l2_normalize(y_pred + tf.keras.backend.epsilon())**2) * tf.nn.l2_normalize(y_pred+ tf.keras.backend.epsilon())**2 )
            return l_ent
        @tf.function
        def sigma_vae_mse(y_true, y_pred):
            """
            MSE loss for sigma-VAE (calibrated decoder).
            """
            log_sigma = tf.math.log(tf.math.sqrt(tf.reduce_mean((y_true - y_pred) ** 2, [0, 1], keepdims=True)))
            
            return tf.reduce_sum(gaussian_nll(y_pred, log_sigma, y_true))

        config = self.config["training"]
        if not self.compiled_model:
            self._compile_model()
        # reset epoch when overwriting history
        if config["overwrite_history"]:
            self.epoch = 0
        self.log.info(f"Training model for {config['epochs']} epochs")
        
        # self.ds.data["train"].get_adata().write("/bmbl_data/jiangyi/SpaGFT/campa_develop/campa_file/tmp/tmp_train.h5ad")
        adata_save = self.ds.data["train"].get_adata()
        import numpy as np
        adata_save.X =  np.array(adata_save.X)
        import scanpy as sc
        from tqdm import tqdm
        assert stored_data_path is not None, "stored_data_path is needed"
        if not use_stored_data and stored_data_path is not None and gft_script_path is not None and gft_env_path is not None:
            for value in tqdm(np.unique(adata_save.obs["mapobject_id"])):
                adata_save[adata_save.obs["mapobject_id"].loc[adata_save.obs["mapobject_id"] ==value ].index].write_h5ad(f"{stored_data_path}/tmp_train_{str(value)}.h5ad")
                success = os.system(f"{gft_env_path}/bin/python {gft_script_path} {stored_data_path}/tmp_train_{str(value)}_FC.h5ad {stored_data_path}/tmp_train_{str(value)}.h5ad")
                if success != 0:
                    print(f"{str(value)} failed")
        else:
            assert gft_script_path is not None, "gft_script_path is needed"
            assert gft_env_path is not None, "gft_env_path is needed"

        # this is the line that actually trains the model, and I will change it to fit SpaGFT
        history_list = []
        x,y = self.ds.get_tf_spagft_use_dataset("train",is_conditional=True)

        for value in range(config["epochs"]):
            history = self.model.model.fit(
                # TODO this is only shuffling the first 10000 samples, but as data is shuffled already should be ok
                x=self.train_dataset.shuffle(10000).batch(config["batch_size"]).prefetch(1),
                validation_data=self.val_dataset.batch(config["batch_size"]).prefetch(1),
                epochs=1,
                verbose=1,
                callbacks=self.callbacks,
            )
            history_c = pd.DataFrame.from_dict(history.history)
            history_c["epoch"] = [value]
            history_list.append(history_c)
            optimizer4fintune = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
            number_iteriter = 0
            for index_id, value_id in enumerate(np.unique(adata_save.obs["mapobject_id"])):
                if os.path.exists(f'{stored_data_path}/tmp_train_{str(value_id)}_FC.h5ad'):
                    adata_read = sc.read_h5ad(f"{stored_data_path}/tmp_train_{str(value_id)}_FC.h5ad")
                    index_read = [int(v) for v in adata_read.obs.index]
                    if max(index_read)<=len(x[0]):
                        input_data = (x[0][index_read],x[1][index_read])
                        number_iteriter+=1
                        with tf.GradientTape() as tape:
                            decoder_output, latent_generate = self.model.model(input_data, training=True)
                            FC_decoder_output = tf.matmul(adata_read.uns["eigvecs"].astype('float32'), decoder_output)
                            FC_latent_generate = tf.matmul(adata_read.uns["eigvecs"].astype('float32'), latent_generate)
                            loss_value =(1-lamda)*sigma_vae_mse(y[index_read], decoder_output)+ lamda * min_entropy(FC_decoder_output,FC_decoder_output)
    
                        grads = tape.gradient(loss_value, self.model.model.trainable_weights)
                        optimizer4fintune.apply_gradients(zip(grads, self.model.model.trainable_weights))
        if config["save_model_weights"]:
            weights_name = self.weights_name.format(self.epoch)
            self.log.info(f"Saving model to {weights_name}")
            self.model.model.save_weights(weights_name)
        history_list = pd.concat(history_list)
        history_list.index = history_list["epoch"]
        return history_list

    def train_model(self):
        """
        Train neural network model.

        Needs an initialised model in ``self.model``.
        """
        self.random_seed(42)
        seed = 42
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1' 
        config = self.config["training"]
        if not self.compiled_model:
            self._compile_model()
        # reset epoch when overwriting history
        if config["overwrite_history"]:
            self.epoch = 0
        self.log.info(f"Training model for {config['epochs']} epochs")
        history = self.model.model.fit(
            # TODO this is only shuffling the first 10000 samples, but as data is shuffled already should be ok
            x=self.train_dataset.shuffle(10000).batch(config["batch_size"]).prefetch(1),
            validation_data=self.val_dataset.batch(config["batch_size"]).prefetch(1),
            epochs=config["epochs"],
            verbose=1,
            callbacks=self.callbacks,
        )
            
        self.epoch += config["epochs"]
        history = pd.DataFrame.from_dict(history.history)
        history["epoch"] = range(self.epoch - config["epochs"], self.epoch)
        if config["save_model_weights"]:
            weights_name = self.weights_name.format(self.epoch)
            self.log.info(f"Saving model to {weights_name}")
            self.model.model.save_weights(weights_name)
        if config["save_history"]:
            if os.path.exists(self.history_name) and not config["overwrite_history"]:
                # if there is a previous history, concatenate to this
                prev_history = pd.read_csv(self.history_name, index_col=0)
                history = pd.concat([prev_history, history])
            history.to_csv(self.history_name)
        return history

    def predict_model(self, data: tf.data.Dataset | np.ndarray, batch_size: int | None = None) -> Any:
        """
        Predict all elements in ``data``.

        Parameters
        ----------
        data
            Data to predict, with first dimension the number of elements.
        batch_size
            Batch size. If None, the training batch size is used.

        Returns
        -------
        ``Iterable``
            prediction
        """
        if isinstance(data, tf.data.Dataset):
            data = data.batch(self.config["training"]["batch_size"])
            batch_size = None
        elif batch_size is None:
            batch_size = self.config["training"]["batch_size"]

        pred = self.model.model.predict(data, batch_size=batch_size)
        if isinstance(pred, list):
            # multiple output model, but only care about first output
            pred = pred[0]
        return pred

    def evaluate_model(self, dataset: tf.data.Dataset | None = None) -> Any:
        """
        Evaluate model using :class:`tf.data.Dataset`.

        Parameters
        ----------
        dataset
            Dataset to evaluate.
            If None, :meth:`Estimator.val_dataset` is used.

        Returns
        -------
        ``Iterable[float]``
            Scores.
        """
        if not self.compiled_model:
            self._compile_model()
        if dataset is None:
            dataset = self.val_dataset
        self.model.model.reset_metrics()
        scores = self.model.model.evaluate(dataset.batch(self.config["training"]["batch_size"]))
        return scores
