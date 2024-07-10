import os
import pandas as pd
import torch
import datetime
from abc import abstractmethod
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


class AbstractTrainer:
    def __init__(self,
                 seq_len: int,
                 batch_size: int,
                 data_path: str,
                 dropout: bool,
                 epochs: int,
                 initialization: str,
                 learning_rate: float,
                 gradient_clipping: float,
                 betas: list[float],
                 initial_model: torch.nn.Module = None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_path = data_path
        self.dropout = dropout
        self.epochs = epochs
        self.initialization = initialization
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping
        self.betas = betas

        self.description = "Abstract Trainer"
        self.model = None
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.optimizer = None
        training_time = datetime.datetime.now()
        self.time = (str(training_time)[:19].
                     replace(":", "-").
                     replace(" ", "_"))

        self.initial_model = initial_model

    @abstractmethod
    def train(self) -> TrainingReview:
        pass

    @abstractmethod
    def as_trainer(self):
        trainer = self.__class__(self.seq_len,
                                 self.batch_size,
                                 self.data_path,
                                 self.dropout,
                                 self.epochs,
                                 self.initialization,
                                 self.learning_rate,
                                 self.gradient_clipping,
                                 self.betas,
                                 self.initial_model)

        trainer.description += f'\nAs trainer {self.time}'
        return trainer


class DefaultTrainer(AbstractTrainer):
    def __init__(self, data_path):
        super().__init__(seq_len=128,
                         batch_size=64,
                         data_path=data_path,
                         dropout=False,
                         epochs=50000,
                         initialization="default",
                         learning_rate=5e-4,
                         gradient_clipping=1.0,
                         betas=[0.9, 0.95])
        self.n_layers = 6
        self.n_heads = 6
        self.embed_size = 192
        self.mlp_hidden_size = self.embed_size * 4
        self.tokenizer, self.data_iter = self._tokenize_data()
        self.model = TransformerLM(
            self.n_layers,
            self.n_heads,
            self.embed_size,
            self.seq_len,
            self.tokenizer.vocab_size(),
            self.mlp_hidden_size,
            with_residuals=True,
            dropout=False
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.learning_rate,
                                     betas=self.betas)

    def _tokenize_data(self):
        os.mkdir(self.time)
        tokenizer, tokenized_data = data.load_data(self.data_path)
        tokenizer.save(self.time + os.sep + "tokenizer.pickle")
        data_iter = iter(data.RandomOrderDataIterator(tokenized_data,
                                                      self.seq_len + 1))
        return tokenizer, data_iter

    def _sample(self):
        self.model.eval()
        sampled = self.tokenizer.detokenize(
            self.model.better_sample_continuation(
                prefix=self.tokenizer.tokenize("Hello"),
                max_tokens_to_generate=500,
                temperature=10,
                topK=5))

        return sampled

    def _save(self, num_batches):
        torch.save(self.model.state_dict(),
                   self.time + os.sep + "model_weights-batch-" + str(num_batches) + ".pth")

    def train(self) -> TrainingReview:
        loss_history = []
        self.model.train()
        num_batches = 0
        while True:
            try:
                for batch in data.batch_items(self.data_iter, self.batch_size):
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
                            sampled = self._sample()
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
                                           index=np.arange(len(loss_history))*10+10,
                                           columns=["loss"]),
                              number_of_epochs=num_batches)

    def as_trainer(self):
        trainer = self.__class__(self.data_path)
        trainer.description += f'\nAs trainer {self.time}'
        return trainer


class Trainer:
    def __init__(self,
                 data_feeder: data.DataFeeder,
                 sampler: transformer.Sampler = None,
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
        self.data_feeder = data_feeder
        self.sampler = sampler

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
                max_context_len=self.data_feeder.seq_len,
                vocab_size=self.data_feeder.tokenizer.vocab_size(),
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

    def train(self) -> TrainingReview:
        loss_history = []
        self.model.train()
        num_batches = 0
        batch_iter = iter(data.batch_items(
            self.data_feeder.data_iter,
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
                            sampled = self._sample()
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
        self.data_feeder.tokenizer.save(self.time + os.sep + "tokenizer.pickle")

    def _sample(self):
        self.model.eval()
        return self.data_feeder.tokenizer.detokenize(self.sampler.sample(self.model))

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
            data_feeder=self.data_feeder,
            sampler=self.sampler,
            initialization="predefined",
            initial_model_path=initial_model_path,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            gradient_clipping=self.gradient_clipping,
            betas=self.betas)

        trainer.description += f'\nAs trainer {self.time}'

        return trainer
