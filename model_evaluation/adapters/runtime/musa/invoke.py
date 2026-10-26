#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from model_evaluation.sdk.manifest import load_manifest
from model_evaluation.sdk.runtime import run_adapter
from impl import OPERATIONS

run_adapter(load_manifest(Path(__file__).with_name("manifest.json")), OPERATIONS)
