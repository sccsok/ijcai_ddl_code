import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.registry import DETECTOR
from .dual_stream import DualStreamDetector
from .mesorch import Mesorch

    
