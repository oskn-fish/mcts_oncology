import os 
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import alpha_zero_bayes
from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
from alpha_zero_bayes.mcts import MCTS