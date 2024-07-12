import os
import pandas as pd
import torch
import datetime
import data
from torch import optim

import transformer
from transformer import TransformerLM
import lm
import numpy as np
import matplotlib.pyplot as plt

import pickle


class TrainingReview:
    def __init__(self,
                 trainer,
                 loss: pd.DataFrame,
                 number_of_epochs: int):
        self.trainer = trainer
        self.loss = loss
        self.number_of_epochs = number_of_epochs

    def plot_loss(self):
        plt.plot(self.loss)

    def get_loss(self):
        return self.loss

    def retrain(self):
        re_trainer = self.trainer.as_trainer()
        return re_trainer.train()

    @staticmethod
    def review_data(reviews: list):
        list_ = [pd.Series({"Training": r.trainer.time,
                            "Final Loss": r.loss.iat[-1],
                            "Number Of Epochs": r.number_of_epochs}
                           ) for r in reviews]
        return pd.DataFrame(list_)

    def save(self, path: str) -> None:
        # TODO: save it.
        with open(path, 'wb') as outfile:
            pickle.dump(self, outfile)


class Trainer:
    def __init__(self,
                 vocab_size: int,
                 max_context_len: int = 128,
                 n_layers: int = 6,
                 n_heads: int = 6,
                 embed_size: int = 192,
                 mlp_hidden_size: int = 768,
                 dropout: bool = False,
                 initialization: str = "default",
                 initial_model_path: str = None,
                 epochs: int = 50000,
                 batch_size: int = 64,
                 learning_rate: float = 5e-4,
                 gradient_clipping: float = 1.0,
                 betas: tuple[float, float] = (0.9, 0.95)):

        self.vocab_size = vocab_size
        self.max_context_len = max_context_len

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.initialization = initialization
        if self.initialization == "predefined":
            self.model = torch.load(initial_model_path)
        else:
            self.model = (TransformerLM(
                n_layers=n_layers,
                n_heads=n_heads,
                embed_size=embed_size,
                max_context_len=self.max_context_len,
                vocab_size=self.vocab_size,
                mlp_hidden_size=mlp_hidden_size,
                with_residuals=True,
                dropout=dropout,
                initialization=self.initialization)
                          .to(self.device))

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping
        self.betas = betas

        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.learning_rate,
                                     betas=self.betas)

        self.description = "AdamW Trainer"

        training_time = datetime.datetime.now()
        self.time = (str(training_time)[:19].
                     replace(":", "-").
                     replace(" ", "_"))

        self._mkdir()

        self._save("initial")

    def train(self, data_feeder, sampler) -> TrainingReview:
        data_feeder.tokenizer.save(self.time + os.sep + "tokenizer.pickle")
        loss_history = []
        self.model.train()
        num_batches = 0
        batch_iter = iter(data.batch_items(
            data_feeder.data_iter,
            self.batch_size))
        while True:
            try:
                for batch in batch_iter:
                    if num_batches >= self.epochs:
                        break
                    num_batches = num_batches + 1

                    batch_x, batch_y = lm.batch_to_labeled_samples(batch)
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    logits = self.model(batch_x)

                    loss = lm.compute_loss(logits, batch_y)

                    # parameters update
                    self.model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping)
                    self.optimizer.step()

                    num_batches += 1
                    if num_batches % 10 == 0:
                        print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                        loss_history.append(loss.item())
                        if num_batches % 100 == 0:
                            sampled = self._sample(data_feeder, sampler)
                            self.model.train()
                            print(f"Model sample: '''{sampled}'''")
                            print("")
                            if num_batches % 500 == 0:
                                self._save(num_batches)
            except KeyboardInterrupt:
                self._save(num_batches)
                print("Interrupted by user -- current weights were saved on batch", num_batches)
                break
        return TrainingReview(self,
                              pd.DataFrame(loss_history,
                                           index=np.arange(len(loss_history)) * 10 + 10,
                                           columns=["loss"]),
                              number_of_epochs=num_batches)

    def _mkdir(self):
        os.mkdir(self.time)

    def _sample(self, data_feeder, sampler):
        self.model.eval()
        return data_feeder.tokenizer.detokenize(sampler.sample(self.model))

    def _save(self, num_batches):
        torch.save(
            self.model.state_dict(),
            self.time
            + os.sep
            + "model_weights-batch-"
            + str(num_batches)
            + ".pth")

    def as_trainer(self):
        initial_model_path = (self.time
                              + os.sep
                              + "model_weights-batch-"
                              + "initial"
                              + ".pth")
        trainer = self.__class__(
            vocab_size=self.vocab_size,
            max_context_len=self.max_context_len,
            initialization="predefined",
            initial_model_path=initial_model_path,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            gradient_clipping=self.gradient_clipping,
            betas=self.betas)

        trainer.description += f'\nAs trainer {self.time}'

        return trainer
    