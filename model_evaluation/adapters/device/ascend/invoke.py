#!/usr/bin/env python3
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT_ROOT))
from model_evaluation.sdk.runtime import run_adapter
from model_evaluation.sdk.manifest import load_manifest
from impl import OPERATIONS
run_adapter(load_manifest(Path(__file__).with_name("manifest.json")), OPERATIONS)
