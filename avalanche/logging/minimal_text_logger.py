from typing import List, TYPE_CHECKING, Tuple, Type, Optional, TextIO
from avalanche.evaluation.metric_utils import stream_type, phase_and_task
if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.metric_results import MetricValue

from avalanche.logging import TextLogger

class MinimalTextLogger(TextLogger):

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super(TextLogger, self).after_training_epoch(strategy, metric_values, **kwargs)
        self.print_current_metrics()
        self.metric_vals = {}

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super(TextLogger, self).after_eval_exp(strategy, metric_values, **kwargs)
        self.print_current_metrics()
        self.metric_vals = {}

    def before_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super(TextLogger, self).before_training(strategy, metric_values, **kwargs)

    def before_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super(TextLogger, self).before_eval(strategy, metric_values, **kwargs)

    def after_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super(TextLogger, self).after_training(strategy, metric_values, **kwargs)

    def after_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super(TextLogger, self).after_eval(strategy, metric_values, **kwargs)
        self.print_current_metrics()
        self.metric_vals = {}

    def _on_exp_start(self, strategy: "SupervisedTemplate"):
        pass
    