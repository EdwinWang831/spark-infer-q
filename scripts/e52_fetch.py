"""Fetch E52 and its existing E43 binary provenance, never model weights."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e50_fetch import main

if __name__=='__main__':main(experiment='e52')
