"""
Quick demo: train a small GPT on TinyShakespeare.
Should reach loss < 2.0 in ~1000 steps on a modern phone.
"""
import sys
sys.path.insert(0, '..')

from engine.trainer import train, Config

# Small config for demo
Config.d_model    = 256
Config.n_head     = 4
Config.n_layer    = 4
Config.block_size = 64
Config.max_steps  = 1000
Config.data_path  = 'data/shakespeare.txt'

train(use_memfile=False)
