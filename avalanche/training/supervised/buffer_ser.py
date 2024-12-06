from collections import defaultdict
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from copy import deepcopy
from torchvision import transforms

from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.templates import SupervisedTemplate


class SER(SupervisedTemplate):
    """
    Implements the SER Strategy,
    from the "Continual Learning with Strong Experience Replay"
    paper, https://arxiv.org/pdf/2305.13622
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        buffer_transforms=transforms.Compose(
            [
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param alpha: float : Hyperparameter weighting the backward consistency loss
        :param beta: float : Hyperparameter weighting the forward consistency loss
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
            **kwargs
        )
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.buffer = Buffer(self.mem_size, self.device, buffer_transforms)
        self.alpha = alpha
        self.beta = beta
        self.old_model = None
        self.is_first_experience = True

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        # Freeze model
        self.old_model = deepcopy(self.model).to(self.device)
        self.old_model.eval()

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        self.is_first_experience = False

    def training_epoch(self, **kwargs):
        """Training epoch.
        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.model(self.mb_x)
            self._after_forward(**kwargs)

            self.loss += F.cross_entropy(
                self.mb_output,
                self.mb_y,
            )

            if not self.is_first_experience:

                buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                    self.batch_size_mem
                )
                buf_inputs, buf_labels, buf_logits = (
                    buf_inputs.to(self.device),
                    buf_labels.to(self.device),
                    buf_logits.to(self.device),
                )

                # Classification loss on memory
                buffer_output = self.model(buf_inputs)

                self.loss += F.cross_entropy(
                    buffer_output,
                    buf_labels,
                )

                # Backward consistency loss on memory data
                self.loss += self.alpha * F.mse_loss(
                    buffer_output,
                    buf_logits.detach(),
                )

                # Forward consistency loss on current task data
                old_mb_output = self.old_model(self.mb_x)
                self.loss += self.beta * F.mse_loss(old_mb_output, self.mb_output)

            self.buffer.add_data(
                examples=self.mb_x, labels=self.mb_y, logits=self.mb_output
            )

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, transforms=lambda x: x):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ["examples", "labels", "logits", "task_labels"]
        self.transforms = transforms

        """
        Initializes just the required tensors.
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith("els") else torch.float32
                setattr(
                    self,
                    attr_str,
                    torch.zeros(
                        (self.buffer_size, *attr.shape[1:]),
                        dtype=typ,
                        device=self.device,
                    ),
                )

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """
        assert size < self.num_seen_examples

        choice = np.random.choice(
            min(self.num_seen_examples, self.examples.shape[0]),
            size=size,
            replace=False,
        )

        ret_tuple = (
            torch.stack([self.transforms(ee.cpu()) for ee in self.examples[choice]]).to(
                self.device
            ),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple


__all__ = ["SER"]
