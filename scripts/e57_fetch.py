"""Archive causal affine screening and actual deferred compressed-decode evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e50_fetch import main

if __name__=='__main__':main(experiment='e57')
