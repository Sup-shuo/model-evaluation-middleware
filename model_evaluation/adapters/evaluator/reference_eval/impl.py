from __future__ import annotations
from pathlib import Path
from model_evaluation.sdk.jsonutil import loads as json_loads
from model_evaluation.sdk.runtime import AdapterError


def requirements(i,c):
    return {
        "schema_version":"1.0",
        "requirements":[
            {"path":"evaluation_environment.python","op":"equals","value":True,
             "message":"reference evaluator requires a Python-capable evaluation environment"}
        ],
    }


def plan_evaluate(i,c):
    task=i['task']; out=Path(i['output_root']).resolve(); runner=Path(__file__).with_name('runner.py').resolve()
    process={
        "schema_version":"1.0",
        "argv":["python",str(runner),"--output-root",str(out),"--task-id",str(task['task_id'])],
        "cwd":str(runner.parent),
        "env_patch":{},
        "stdin":{"mode":"null"},
        "stdout":{"mode":"file","path":str(Path(i.get('log_path') or out/'evaluation.log'))},
        "stderr":{"mode":"merge_stdout"},
        "timeout_seconds":30,
        "metadata":{"purpose":"adapter-protocol-reference"},
    }
    return {"process":process,"raw_result_root":str(out)}


def normalize(i,c):
    root=Path(i['raw_result_root']).resolve(); p=root/'reference_result.json'
    if not p.is_file(): raise AdapterError('RESULT_INVALID',f'reference result missing: {p}')
    obj=json_loads(p.read_text(encoding='utf-8'))
    if not isinstance(obj,dict) or not isinstance(obj.get('metrics'),dict):
        raise AdapterError('RESULT_INVALID','reference result must contain metrics object')
    run=i['run_metadata']; metrics={}
    for key,value in obj['metrics'].items(): metrics[str(key)]={"value":value}
    return {
        "schema_version":"1.0",
        "run_id":run['run_id'],
        "model":run['model'],
        "benchmark":run['benchmark'],
        "framework":"reference_eval",
        "metrics":metrics,
        "raw_result":{"path":str(p),"media_type":"application/json"},
        "metadata":{"purpose":"adapter-protocol-reference","task_id":obj.get('task_id')},
    }


def snapshot(i,c):
    return {"framework":"reference_eval","purpose":"adapter-protocol-reference"}


OPERATIONS={"requirements":requirements,"plan_evaluate":plan_evaluate,"normalize":normalize,"snapshot":snapshot}
