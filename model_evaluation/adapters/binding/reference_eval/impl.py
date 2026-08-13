from __future__ import annotations
import hashlib, json


def requirements(i,c):
    return {"schema_version":"1.0","requirements":[]}


def _basis(benchmark,dataset_artifact):
    return {
        "binding":"reference_eval/v1",
        "benchmark_id":benchmark['id'],
        "benchmark_protocol":benchmark.get('protocol') or {},
        "dataset_fingerprint":dataset_artifact.get('fingerprint'),
    }


def _fingerprint(benchmark,dataset_artifact):
    raw=json.dumps(_basis(benchmark,dataset_artifact),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()


def build_task(i,c):
    b=i['benchmark']; d=i['dataset_artifact']; fp=_fingerprint(b,d)
    return {
        "schema_version":"1.0",
        "framework":"reference_eval",
        "benchmark_id":b['id'],
        "task_id":b['id'],
        "protocol_fingerprint":fp,
        "metadata":{"purpose":"adapter-protocol-reference","dataset_fingerprint":d.get('fingerprint')},
    }


def fingerprint(i,c):
    return {"protocol_fingerprint":_fingerprint(i['benchmark'],i['dataset_artifact'])}


OPERATIONS={"requirements":requirements,"build_task":build_task,"protocol_fingerprint":fingerprint}
