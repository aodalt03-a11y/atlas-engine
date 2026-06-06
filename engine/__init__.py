"""
Atlas Engine
ARM64-optimized LLM training engine for Android/Termux.
"""
from .model import GPT
from .trainer import train, Config
from .tensor import Tensor
