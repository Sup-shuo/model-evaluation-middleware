#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root',required=True)
    ap.add_argument('--task-id',required=True)
    args=ap.parse_args()
    root=Path(args.output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    (root/'reference_result.json').write_text(json.dumps({
        'task_id':args.task_id,
        'metrics':{'contract_ok':1}
    },sort_keys=True)+'\n',encoding='utf-8')

if __name__=='__main__': main()
