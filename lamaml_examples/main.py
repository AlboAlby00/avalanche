import torch
from os.path import expanduser
import argparse
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST
from avalanche.logging import InteractiveLogger, TextLogger, MinimalTextLogger
from avalanche.training.supervised import Naive, EWC, Replay
from avalanche.training.supervised.lamaml_v2 import LaMAML
from torchvision import datasets
import torchvision.transforms as transforms

USE_GPU = True

class TaskIgnoringSimpleMLP(SimpleMLP):
    def forward(self, x, t=None):
        return super().forward(x)


def main():
    parser = argparse.ArgumentParser(description="Choose strategy and set parameters for continual learning.")
    
    # General arguments
    parser.add_argument('--strategy', choices=['ewc', 'naive', 'lamaml', 'replay'], default="lamaml", help="Select the strategy")
    parser.add_argument('--benchmark', choices=['permuted', 'rotated'], default="permuted", help="Select the benchmark")
    parser.add_argument('--train_epochs', type=int, default=1, help="Number of training epochs per experience")
    parser.add_argument('--train_mb_size', type=int, default=10, help="Mini-batch size for training")
    parser.add_argument('--eval_mb_size', type=int, default=16, help="Mini-batch size for evaluation")
    parser.add_argument('--lr', type=float, default=0.3, help="Learning rate for the optimizer")
    parser.add_argument('--hidden_size', type=int, default=100, help="Hidden layer size for the model")
    parser.add_argument('--hidden_layers', type=int, default=2, help="Number of hidden layers in the model")
    parser.add_argument('--drop_rate', type=float, default=0.0, help="Dropout rate in the model")
    parser.add_argument('--num_train_images_per_task', type=int, default=1000, help="Number of images per task used during training")
    parser.add_argument('--num_test_images_per_task', type=float, default=100, help="Number of images per task used during evaluation")
    parser.add_argument('--num_tasks', type=int, default=20, help="Total number of tasks")
    parser.add_argument('--seed', type=int, default=42, help="Total number of tasks")
    
    # Strategy-specific arguments
    parser.add_argument('--ewc_lambda', type=float, default=0.4, help="EWC regularization strength (EWC only)")
    parser.add_argument('--mem_size', type=int, default=200, help="Replay memory size (Replay only)")
    parser.add_argument('--buffer_mb_size', type=int, default=10, help="Buffer size for LaMAML")
    parser.add_argument('--alpha_init', type=float, default=0.15, help="Initial alpha value for LaMAML")
    parser.add_argument('--lr_alpha', type=float, default=0.3, help="Learning rate for alpha in LaMAML")
    parser.add_argument('--grad_clip_norm', type=float, default=2.0, help="Gradient clipping norm for LaMAML")
    parser.add_argument('--max_buffer_size', type=int, default=200, help="Maximum buffer size for LaMAML")
    parser.add_argument('--n_inner_updates', type=int, default=5, help="Number of inner updates for LaMAML")
    parser.add_argument('--use_second_order', action='store_true', default=False, help="Calculate second order terms in LaMAML")
    parser.add_argument('--sync_update', action='store_true', default=False, help="Calculate second order terms in LaMAML")
    parser.add_argument('--learn_lr', action='store_true', default=False, help="Select CMaml or LaMaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
    print(f"Using device: {device}")

    # Create the benchmark
    if args.benchmark == "permuted":
        benchmark = PermutedMNIST(
            n_experiences=args.num_tasks, dataset_root=expanduser("~") + "/.avalanche/data/mnist/", seed=args.seed, num_train_images_per_task=args.num_train_images_per_task, 
            num_test_images_per_task=args.num_test_images_per_task
            # train_transform=transforms.Lambda(lambda x: x), eval_transform=transforms.Lambda(lambda x: x)
        )
    elif args.benchmark == "rotated":
        benchmark = RotatedMNIST(
            n_experiences=20, dataset_root=expanduser("~") + "/.avalanche/data/mnist/", seed=args.seed, num_images_per_task=args.num_images_per_task, 
            # train_transform=transforms.Lambda(lambda x: x), eval_transform=transforms.Lambda(lambda x: x)
        )

    # Choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger()
    minimal_text_logger = MinimalTextLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=False, experience=False,
                         trained_experience=False, stream=True),
        loggers=[minimal_text_logger],
    )

    model = TaskIgnoringSimpleMLP(hidden_size=100, hidden_layers=2, drop_rate=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    criterion = torch.nn.CrossEntropyLoss()

    # Create the strategy based on selected argument
    # Model setup
    model = TaskIgnoringSimpleMLP(
        hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, drop_rate=args.drop_rate
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Create strategy based on the selected argument
    if args.strategy == 'lamaml':
        strategy = LaMAML(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=args.train_epochs,
            device=device,
            train_mb_size=args.train_mb_size,
            eval_mb_size=args.eval_mb_size,
            buffer_mb_size=args.buffer_mb_size,
            evaluator=eval_plugin,
            second_order=args.use_second_order,
            grad_clip_norm=args.grad_clip_norm,
            alpha_init=args.alpha_init,
            lr_alpha=args.lr_alpha,
            sync_update=args.sync_update,
            max_buffer_size=args.max_buffer_size,
            learn_lr=args.learn_lr,
            n_inner_updates=args.n_inner_updates,
        )
    elif args.strategy == 'ewc':
        strategy = EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=args.train_epochs,
            device=device,
            train_mb_size=args.train_mb_size,
            eval_mb_size=args.eval_mb_size,
            evaluator=eval_plugin,
            ewc_lambda=args.ewc_lambda,
        )
    elif args.strategy == 'naive':
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=args.train_epochs,
            device=device,
            train_mb_size=args.train_mb_size,
            eval_mb_size=args.eval_mb_size,
            evaluator=eval_plugin,
        )
    elif args.strategy == 'replay':
        strategy = Replay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=args.train_epochs,
            device=device,
            train_mb_size=args.train_mb_size,
            eval_mb_size=args.eval_mb_size,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
        )

    # Train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    main()
