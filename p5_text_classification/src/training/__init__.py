"""
Training and evaluation module.
Implements the Single Responsibility Principle (SRP) and Dependency Inversion Principle (DIP).
"""

from .trainer_interface import TrainerInterface, EvaluatorInterface
from .trainer import Trainer
from .evaluator import ClassificationEvaluator

__all__ = [
    'TrainerInterface', 
    'EvaluatorInterface', 
    'Trainer', 
    'ClassificationEvaluator',
] 