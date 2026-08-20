from __future__ import annotations
import copy, json, os, shutil, socket, sys, tempfile, time, unittest, subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
PACKAGE_ROOT=ROOT/'model_evaluation'
sys.path.insert(0,str(ROOT))
from model_evaluation.core.app import Application
from model_evaluation.core.compatibility import evaluate, facts_from_runtime, facts_from_environment, facts_from_service
from model_evaluation.core.errors import AdapterExecutionError, CompatibilityError, ConfigError, ProcessError, ResourceError
from model_evaluation.core.process.env import EnvPatchMerger, derive_env_patch_additions, prepare_process_for_environment
from model_evaluation.core.process.manager import ProcessManager, SecretStore
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.security import adapter_subprocess_env, execution_subprocess_env
from model_evaluation.core.config.loader import _reject_inline_secrets, _apply_spec_defaults
from model_evaluation.core.config.parsing import load_yaml_strict
from model_evaluation.core.serialization import json_dumps_strict, json_loads_strict
from model_evaluation.sdk.jsonutil import dumps as adapter_json_dumps, loads as adapter_json_loads
from model_evaluation.sdk.http import _NoRedirect
from model_evaluation.core.config.deployment import resolve_deployment_profile
from model_evaluation.core.config.evaluation import resolve_evaluation_profile
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.config.overrides import validate_run_overrides
from model_evaluation.core.matrix import finalize_matrix_plan, verify_matrix_plan, MatrixSchemas, MatrixRepository, MatrixPlanner, MatrixExecutor
from model_evaluation.core.result_relocation import load_result_relocation
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.registry.operation_contracts import validate_operation_input, validate_operation_output
from model_evaluation.core.registry.adapter_registry import _validate_schema_versions, _validate_adapter_name
from model_evaluation.core.orchestrator import Orchestrator
from model_evaluation.core.provenance import assess_model_provenance
from model_evaluation.sdk.runtime import AdapterError

def _load_mlu_impl_for_test():
    import importlib.util
    path=PACKAGE_ROOT / "adapters" / "device" / "mlu" / "impl.py"
    spec=importlib.util.spec_from_file_location("_mlu_impl_test", path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _load_adapter_impl_for_test(kind: str, name: str):
    import importlib.util
    path=PACKAGE_ROOT / "adapters" / kind / name / "impl.py"
    module_name="_adapter_test_"+kind+"_"+name.replace('.','_')
    spec=importlib.util.spec_from_file_location(module_name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

class CoreTests(unittest.TestCase):
    def setUp(self): self.schemas=SchemaStore(PACKAGE_ROOT/'schemas')
    def test_mlu_device_node_discovery_ignores_ipcm_and_control_nodes(self):
        mod=_load_mlu_impl_for_test()
        nodes=[
            "/dev/cambricon_ctl", "/dev/cambricon_gdr",
            "/dev/cambricon_dev0", "/dev/cambricon_dev1", "/dev/cambricon_dev7",
            "/dev/cambricon_ipcm0", "/dev/cambricon_ipcm1", "/dev/cambricon_ipcm7",
        ]
        self.assertEqual(mod._ids_from_nodes(nodes), ["0", "1", "7"])

    def test_metax_device_adapter_parses_inventory_and_owns_visibility(self):
        mod=_load_adapter_impl_for_test("device", "metax")
        listing=(
            "mx-smi  version: 2.3.4\n"
            "GPU#0    MXC500      0000:08:00.0   Available "
            "(UUID: GPU-test-0)\n"
            "GPU#2    MXC500      0000:0e:00.0   Available "
            "(UUID: GPU-test-2)\n"
        )
        with patch.object(mod.shutil, "which", return_value="/usr/bin/mx-smi"):
            with patch.object(
                mod,
                "_run",
                return_value=SimpleNamespace(returncode=0, stdout=listing, stderr=""),
            ):
                devices=mod._listed_devices(1)
        self.assertEqual([item["id"] for item in devices], ["0", "2"])
        self.assertEqual(devices[0]["name"], "MetaX MXC500")
        self.assertEqual(devices[0]["uuid"], "GPU-test-0")
        self.assertEqual(
            mod.visibility({"devices": [2]}, {}),
            {"env_patch": {"set": {"MACA_VISIBLE_DEVICES": "2"}}},
        )

    def test_maca_runtime_patch_is_explicit_and_machine_owned(self):
        mod=_load_adapter_impl_for_test("runtime", "maca")
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"maca"
            driver=Path(td)/"driver"
            for path in (
                root/"bin",
                root/"lib",
                root/"mxgpu_llvm"/"bin",
                root/"mxshmem"/"lib",
                driver/"bin",
                driver/"lib",
            ):
                path.mkdir(parents=True,exist_ok=True)
            patch_value=mod.resolve_environment(
                {"parameters":{"root":str(root),"driver_root":str(driver)}},
                {},
            )["env_patch"]
        self.assertEqual(patch_value["set"]["MACA_PATH"],str(root.resolve()))
        self.assertEqual(patch_value["set"]["MACA_HOME"],str(root.resolve()))
        self.assertEqual(patch_value["set"]["PYTORCH_NVML_BASED_CUDA_CHECK"],"1")
        self.assertIn(str((root/"bin").resolve()),patch_value["prepend_path"]["PATH"])
        self.assertIn(str((driver/"lib").resolve()),patch_value["prepend_path"]["LD_LIBRARY_PATH"])

    def test_schema_discovery_ignores_macos_metadata_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=PACKAGE_ROOT/'schemas'/'adapter_manifest.schema.json'
            shutil.copy2(source,root/source.name)
            (root/'._adapter_manifest.schema.json').write_bytes(b'\x00AppleDouble')
            checked=SchemaStore(root).validate_all_schemas()
            self.assertEqual(checked,[source.name])

    def test_bbh_basic_integrity_does_not_hash_or_require_official_sample_count(self):
        from unittest.mock import patch
        mod=_load_adapter_impl_for_test('dataset','bbh_local')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'task.json'
            path.write_text(json.dumps({'examples':[{'input':'question','target':'answer'}]}))
            with patch.object(mod,'_sha',side_effect=AssertionError('basic must not hash dataset bytes')):
                mod._validate_file(path,250,'0'*64,integrity_policy='basic')
        self.assertEqual(mod._integrity_policy({'dataset':{'parameters':{}}}),'basic')

    def test_bbh_strict_integrity_rejects_digest_or_sample_mismatch(self):
        mod=_load_adapter_impl_for_test('dataset','bbh_local')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'task.json'
            path.write_text(json.dumps({'examples':[{'input':'question','target':'answer'}]}))
            with self.assertRaisesRegex(AdapterError,'sample count mismatch'):
                mod._validate_file(path,2,'0'*64,integrity_policy='strict')
            with self.assertRaisesRegex(AdapterError,'SHA256 mismatch'):
                mod._validate_file(path,1,'0'*64,integrity_policy='strict')

    def test_local_files_basic_omits_external_byte_digests(self):
        from unittest.mock import patch
        mod=_load_adapter_impl_for_test('dataset','local_files')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'data.json').write_text('[{"x":1}]')
            benchmark={'id':'custom','dataset':{'revision':'declared','parameters':{'root':str(root),'files':['data.json']}}}
            with patch.object(mod,'_sha',side_effect=AssertionError('basic must not hash dataset bytes')):
                artifact=mod._build(benchmark)
        self.assertNotIn('fingerprint',artifact); self.assertNotIn('sha256',artifact['files'][0])
        self.assertEqual(artifact['metadata']['integrity_policy'],'basic')
        self.assertEqual(artifact['metadata']['revision_provenance'],'user-declared')

    def test_device_adapter_owns_default_device_selection(self):
        from unittest.mock import patch
        from model_evaluation.adapters.device.nvidia import impl as nvidia_impl
        with patch.object(nvidia_impl,'_rows',return_value=[['4','GPU A','1024','u4'],['7','GPU B','2048','u7']]):
            desc=nvidia_impl.probe({'requested_devices':[]},{'timeout_seconds':1})
        self.assertEqual([d['id'] for d in desc['devices']],['4'])

    def test_requirements(self):
        r={"schema_version":"1.0","requirements":[{"path":"runtime.family","op":"in","value":["cuda","rocm"]}]}; self.schemas.validate('requirement_set',r); self.assertTrue(evaluate(r,{"runtime.family":"cuda"}).compatible); self.assertFalse(evaluate(r,{"runtime.family":"cpu"}).compatible)
    def test_unknown_version_never_satisfies_min_version(self):
        r={"schema_version":"1.0","requirements":[{"path":"runtime.version","op":"min_version","value":"12.0"}]}
        self.assertFalse(evaluate(r,{"runtime.version":"unknown"}).compatible); self.assertTrue(evaluate(r,{"runtime.version":"12.4"}).compatible)
    def test_capability_cannot_override_structural_fact(self):
        d={"family":"cuda","version":"12.4","available":True,"capabilities":{"values":{"runtime.family":"rocm"}}}
        with self.assertRaises(CompatibilityError): facts_from_runtime(d)
    def test_env_ownership_conflict(self):
        m=EnvPatchMerger(); m.add('device',{'set':{'X':'1'}})
        with self.assertRaises(ConfigError): m.add('backend',{'set':{'X':'2'}})
    def test_path_mutations_are_composable(self):
        m=EnvPatchMerger(); m.add('runtime',{'prepend_path':{'PATH':['/runtime/bin']}}); m.add('environment',{'prepend_path':{'PATH':['/env/bin']}})
        self.assertEqual(m.result()['prepend_path']['PATH'],['/runtime/bin','/env/bin'])
    def test_process_spec_shape(self): self.schemas.validate('process_spec',{"schema_version":"1.0","argv":["python","-V"],"env_patch":{}})
    def test_port_resource_host_is_structured(self):
        claim={"kind":"port","id":"8091","exclusive":True,"host":"127.0.0.1"}
        self.schemas.validate('resource_claim',claim)
        self.assertNotIn('metadata',claim)

    def test_port_check_rejects_listener_but_allows_serial_reuse_after_time_wait(self):
        host = "127.0.0.1"
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.addCleanup(listener.close)
        listener.bind((host, 0))
        port = int(listener.getsockname()[1])
        listener.listen()
        with self.assertRaisesRegex(ResourceError, "port unavailable"):
            ResourceManager.check_port(host, port)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(client.close)
        client.connect((host, port))
        accepted, _ = listener.accept()
        # The server side closes first, placing this local port in TIME_WAIT
        # once the client observes EOF and closes.  A following managed server
        # with SO_REUSEADDR can bind immediately and so must our availability
        # check.
        accepted.close()
        self.assertEqual(client.recv(1), b"")
        client.close()
        listener.close()
        ResourceManager.check_port(host, port)
        successor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(successor.close)
        successor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        successor.bind((host, port))
        successor.listen(1)
    def test_inline_secret_rejected(self):
        with self.assertRaises(ConfigError): _reject_inline_secrets({'api_key':'cleartext'})
        _reject_inline_secrets({'auth':{'secret_ref':'secret://env/MODEL_API_KEY'}})
    def test_adapter_environment_drops_secrets(self):
        env=adapter_subprocess_env({'PATH':'/bin','HOME':'/tmp','MODEL_API_KEY':'secret','HF_TOKEN':'secret2','NEUWARE_HOME':'/opt/n','CUDA_HOME':'/opt/cuda'})
        self.assertEqual(env['PATH'],'/bin'); self.assertNotIn('NEUWARE_HOME',env); self.assertNotIn('CUDA_HOME',env); self.assertNotIn('MODEL_API_KEY',env); self.assertNotIn('HF_TOKEN',env)
    def test_environment_wrapper_additions(self):
        before={'set':{'A':'1'},'prepend_path':{'PATH':['/a']}}; after={'set':{'A':'1','B':'2'},'prepend_path':{'PATH':['/b','/a']}}
        add=derive_env_patch_additions(before,after); self.assertEqual(add['set'],{'B':'2'}); self.assertEqual(add['prepend_path'],{'PATH':['/b']})

    def test_environment_wrapper_keeps_selected_environment_path_first(self):
        process={'schema_version':'1.0','argv':['tool'],'env_patch':{'prepend_path':{'PATH':['/backend/bin']}}}
        def wrap(prepared):
            out=copy.deepcopy(prepared); patch=copy.deepcopy(out.get('env_patch') or {})
            existing=list((patch.get('prepend_path') or {}).get('PATH') or [])
            patch.setdefault('prepend_path',{})['PATH']=['/selected-env/bin',*existing]
            out['env_patch']=patch; return out
        wrapped=prepare_process_for_environment(
            process,base_patches=(('runtime',{'prepend_path':{'PATH':['/runtime/bin']}}),),
            process_owner='backend',wrap=wrap,
        )
        self.assertEqual(wrapped['env_patch']['prepend_path']['PATH'],['/selected-env/bin','/runtime/bin','/backend/bin'])

    def test_orchestrator_environment_preparation_matches_selected_venv_precedence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'venv'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            tool=bindir/'tool'; tool.write_text('#!/bin/sh\nexit 0\n'); tool.chmod(0o755)
            app=Application(PACKAGE_ROOT, ROOT); orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
            env_client=app.registry.get('environment','venv')
            env_desc=env_client.invoke('resolve',{'profile':str(env_root),'parameters':{}},context={},timeout=2)
            platform={
                'backend_environment':{'provider':'venv','profile':str(env_root)},
                'evaluation_environment':{'provider':'current','profile':'current'},
            }
            resolved={'backend_environment':env_desc}
            process={'schema_version':'1.0','argv':['tool'],'env_patch':{'prepend_path':{'PATH':['/backend/bin']}}}
            wrapped,_=orch.prepare_process_for_environment(
                process,platform_spec=platform,resolved_platform=resolved,role='backend',
                base_patches=(('runtime',{'prepend_path':{'PATH':['/runtime/bin']}}),),
                context={'test':True},timeout=2,
            )
            self.assertEqual(wrapped['env_patch']['prepend_path']['PATH'],[str(bindir),'/runtime/bin','/backend/bin'])

    def test_backend_start_plan_dependency_probe_is_formal_contract(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); model=root/'model'; model.mkdir()
            inp={
                'model':{'id':'m'},
                'deployment':{'management':{'mode':'managed'},'model_location':{'local_path':str(model)},'parameters':{'port':8091}},
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},'log_path':str(root/'backend.log'),'network_policy':'offline',
            }
            out=vllm_impl.plan_start(inp,{'offline':True})
            validate_operation_output(self.schemas,'backend','plan_start',out,input_obj=inp)
            self.assertEqual(out['schema_version'],'1.1')
            self.assertEqual(out['dependency_probe']['argv'][:2],['vllm','--version'])
            self.assertEqual(out['dependency_probe']['timeout_seconds'],45.0)
            self.assertEqual(out['shutdown'],{'strategy':'signal','signal':'SIGTERM','timeout_seconds':15.0})
            self.assertNotIn('version_argv',out['process'].get('metadata') or {})

    def test_vllm_preflight_plan_has_portable_dependency_and_model_phases(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model'; model.mkdir()
            inp={
                'model':{
                    'schema_version':'1.0','id':'m',
                    'source':{'type':'local','ref':str(model)},
                    'quantization':'awq','context_length':8192,
                },
                'deployment':{
                    'schema_version':'1.1','id':'vllm-managed','backend':{'adapter':'vllm'},
                    'management':{'mode':'managed'},
                    'model_location':{'local_path':str(model)},
                    'parameters':{'tensor_parallel_size':2,'dependency_probe_timeout_seconds':33,'model_probe_timeout_seconds':77},
                },
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},
                'log_path':str(Path(td)/'backend.log'),'network_policy':'offline',
            }
            out=vllm_impl.plan_preflight(inp,{'offline':True})
            validate_operation_input(self.schemas,'backend','plan_preflight',{
                'model':inp['model'],'deployment':inp['deployment'],'platform':inp['platform'],'network_policy':'offline',
            })
            validate_operation_output(self.schemas,'backend','plan_preflight',out,input_obj=inp)
            self.assertEqual([p['phase'] for p in out['probes']],['backend_dependency','model_compatibility'])
            self.assertEqual(out['probes'][0]['process']['timeout_seconds'],33)
            self.assertEqual(out['probes'][1]['process']['timeout_seconds'],77)
            self.assertEqual(out['probes'][1]['result_format'],'preflight_result')
            payload=json_loads_strict(out['probes'][1]['process']['argv'][-1])
            self.assertEqual(payload['quantization'],'awq')
            self.assertEqual(payload['max_model_len'],8192)
            self.assertEqual(payload['tensor_parallel_size'],2)

    def test_vllm_language_model_only_is_adapter_owned_and_reaches_preflight(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model'; model.mkdir()
            inp={
                'model':{'id':'m'},
                'deployment':{
                    'management':{'mode':'managed'},
                    'model_location':{'local_path':str(model)},
                    'parameters':{'port':8091,'language_model_only':True},
                },
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},
                'log_path':str(Path(td)/'backend.log'),'network_policy':'offline',
            }
            start=vllm_impl.plan_start(inp,{'offline':True})
            self.assertEqual(start['process']['argv'].count('--language-model-only'),1)
            preflight=vllm_impl.plan_preflight(inp,{'offline':True})
            payload=json_loads_strict(preflight['probes'][1]['process']['argv'][-1])
            self.assertIs(payload['language_model_only'],True)
            inp['deployment']['parameters']['extra_args']=['--language-model-only']
            with self.assertRaisesRegex(Exception,'adapter-owned field --language-model-only'):
                vllm_impl.plan_start(inp,{'offline':True})

    def test_vllm_managed_process_uses_run_workspace_not_model_parent_as_cwd(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            model=root/'object-backed-models'/'Org'/'Model'
            workspace=root/'results'/'run-id'
            model.mkdir(parents=True); workspace.mkdir(parents=True)
            inp={
                'model':{'id':'m'},
                'deployment':{
                    'management':{'mode':'managed'},
                    'model_location':{'local_path':str(model)},
                    'parameters':{'port':8091},
                },
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},
                'log_path':str(workspace/'backend.log'),'network_policy':'offline',
            }
            start=vllm_impl.plan_start(
                inp,{'offline':True,'workspace':str(workspace)}
            )
            self.assertEqual(start['process']['cwd'],str(workspace))
            self.assertNotEqual(start['process']['cwd'],str(model.parent))
            inp['deployment']['parameters']['cwd']=str(root/'explicit')
            explicit=vllm_impl.plan_start(
                inp,{'offline':True,'workspace':str(workspace)}
            )
            self.assertEqual(explicit['process']['cwd'],str(root/'explicit'))

    def test_backend_preflight_contract_rejects_empty_or_misordered_dependency_phase(self):
        process={'schema_version':'1.0','argv':['true']}
        only_model={'schema_version':'1.0','probes':[
            {'id':'model.config','phase':'model_compatibility','required':True,'result_format':'text','process':process},
        ]}
        with self.assertRaisesRegex(Exception,'required backend_dependency'):
            validate_operation_output(self.schemas,'backend','plan_preflight',only_model,input_obj={})
        misordered={'schema_version':'1.0','probes':[
            {'id':'model.config','phase':'model_compatibility','required':True,'result_format':'text','process':process},
            {'id':'backend.import','phase':'backend_dependency','required':True,'result_format':'text','process':process},
        ]}
        with self.assertRaisesRegex(Exception,'must precede'):
            validate_operation_output(self.schemas,'backend','plan_preflight',misordered,input_obj={})

    def test_backend_preflight_report_is_structured_and_blocks_model_after_dependency_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'venv'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            app=Application(PACKAGE_ROOT, ROOT); orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
            env_client=app.registry.get('environment','venv')
            env_desc=env_client.invoke('resolve',{'profile':str(env_root),'parameters':{}},context={},timeout=2)
            platform={
                'backend_environment':{'provider':'venv','profile':str(env_root)},
                'evaluation_environment':{'provider':'current','profile':'current'},
            }
            resolved={'backend_environment':env_desc}
            process=lambda code:{'schema_version':'1.0','argv':['python','-c',code],'stdin':{'mode':'null'},'stdout':{'mode':'capture'},'stderr':{'mode':'capture'},'timeout_seconds':2,'metadata':{}}
            plan={'schema_version':'1.0','probes':[
                {'id':'backend.import','phase':'backend_dependency','required':True,'result_format':'text','process':process('import sys;sys.exit(9)')},
                {'id':'model.config','phase':'model_compatibility','required':True,'result_format':'preflight_result','process':process('print(\'{"schema_version":"1.0","status":"passed","facts":{"ok":true}}\')')},
            ]}
            report=orch.run_backend_preflight(plan,platform_spec=platform,resolved_platform=resolved,raise_on_failure=False)
            self.schemas.validate('preflight_report',report)
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['probes'][0]['status'],'failed')
            self.assertEqual(report['probes'][1]['status'],'blocked')

    def test_backend_preflight_parses_last_json_line(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'venv'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            app=Application(PACKAGE_ROOT, ROOT); orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
            env_client=app.registry.get('environment','venv')
            env_desc=env_client.invoke('resolve',{'profile':str(env_root),'parameters':{}},context={},timeout=2)
            platform={'backend_environment':{'provider':'venv','profile':str(env_root)},'evaluation_environment':{'provider':'current','profile':'current'}}
            process={'schema_version':'1.0','argv':['python','-c','print("noise");print(\'{"schema_version":"1.0","status":"passed","facts":{"platform":"portable","weights_loaded":false}}\')'],'stdin':{'mode':'null'},'stdout':{'mode':'capture'},'stderr':{'mode':'capture'},'timeout_seconds':2,'metadata':{}}
            plan={'schema_version':'1.0','probes':[{'id':'model.config','phase':'model_compatibility','required':True,'result_format':'preflight_result','process':process}]}
            report=orch.run_backend_preflight(plan,platform_spec=platform,resolved_platform={'backend_environment':env_desc},raise_on_failure=False)
            self.assertEqual(report['status'],'passed')
            self.assertEqual(report['probes'][0]['result']['facts']['platform'],'portable')
            self.assertFalse(report['probes'][0]['result']['facts']['weights_loaded'])

    def test_backend_preflight_captures_structured_domain_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'venv'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            app=Application(PACKAGE_ROOT, ROOT); orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
            env_desc=app.registry.get('environment','venv').invoke('resolve',{'profile':str(env_root),'parameters':{}},context={},timeout=2)
            platform={'backend_environment':{'provider':'venv','profile':str(env_root)},'evaluation_environment':{'provider':'current','profile':'current'}}
            result='{"schema_version":"1.0","status":"failed","error":{"code":"MODEL_CONFIG_INCOMPATIBLE","message":"missing quantization metadata"}}'
            process={'schema_version':'1.0','argv':['python','-c',f'import sys;print({result!r});sys.exit(2)'],'stdin':{'mode':'null'},'stdout':{'mode':'capture'},'stderr':{'mode':'capture'},'timeout_seconds':2,'metadata':{}}
            plan={'schema_version':'1.0','probes':[{'id':'model.config','phase':'model_compatibility','required':True,'result_format':'preflight_result','process':process}]}
            report=orch.run_backend_preflight(plan,platform_spec=platform,resolved_platform={'backend_environment':env_desc},raise_on_failure=False)
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['probes'][0]['result']['error']['code'],'MODEL_CONFIG_INCOMPATIBLE')
            self.assertIn('missing quantization metadata',report['probes'][0]['error'])

    def test_backend_preflight_rejects_process_result_status_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'venv'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            app=Application(PACKAGE_ROOT, ROOT); orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
            env_desc=app.registry.get('environment','venv').invoke('resolve',{'profile':str(env_root),'parameters':{}},context={},timeout=2)
            platform={'backend_environment':{'provider':'venv','profile':str(env_root)},'evaluation_environment':{'provider':'current','profile':'current'}}
            result='{"schema_version":"1.0","status":"failed","error":{"code":"PROBE_REJECTED","message":"domain failure"}}'
            process={'schema_version':'1.0','argv':['python','-c',f'print({result!r})'],'stdin':{'mode':'null'},'stdout':{'mode':'capture'},'stderr':{'mode':'capture'},'timeout_seconds':2,'metadata':{}}
            plan={'schema_version':'1.0','probes':[{'id':'backend.protocol','phase':'backend_dependency','required':True,'result_format':'preflight_result','process':process}]}
            report=orch.run_backend_preflight(plan,platform_spec=platform,resolved_platform={'backend_environment':env_desc},raise_on_failure=False)
            self.assertEqual(report['status'],'failed')
            self.assertIn('status mismatch',report['probes'][0]['error'])

    def test_backend_preflight_applies_portable_device_and_runtime_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'venv'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            app=Application(PACKAGE_ROOT, ROOT); orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
            env_desc=app.registry.get('environment','venv').invoke('resolve',{'profile':str(env_root),'parameters':{}},context={},timeout=2)
            platform={'backend_environment':{'provider':'venv','profile':str(env_root)},'evaluation_environment':{'provider':'current','profile':'current'}}
            code=("import json,os;print(json.dumps({'schema_version':'1.0','status':'passed','facts':"
                  "{'device':os.environ.get('PROBE_DEVICE_TOKEN'),'runtime':os.environ.get('PROBE_RUNTIME_TOKEN')}}))")
            process={'schema_version':'1.0','argv':['python','-c',code],'stdin':{'mode':'null'},'stdout':{'mode':'capture'},'stderr':{'mode':'capture'},'timeout_seconds':2,'metadata':{}}
            resolved={
                'backend_environment':env_desc,
                'device_env_patch':{'set':{'PROBE_DEVICE_TOKEN':'third-device'}},
                'runtime_env_patch':{'set':{'PROBE_RUNTIME_TOKEN':'portable-runtime'}},
            }
            plan={'schema_version':'1.0','probes':[{'id':'backend.environment','phase':'backend_dependency','required':True,'result_format':'preflight_result','process':process}]}
            report=orch.run_backend_preflight(plan,platform_spec=platform,resolved_platform=resolved,raise_on_failure=False)
            facts=report['probes'][0]['result']['facts']
            self.assertEqual(facts,{'device':'third-device','runtime':'portable-runtime'})

    def test_managed_backend_start_plan_requires_shutdown_contract(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); model=root/'model'; model.mkdir()
            inp={
                'model':{'id':'m'},
                'deployment':{'management':{'mode':'managed'},'model_location':{'local_path':str(model)},'parameters':{'port':8091}},
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},'log_path':str(root/'backend.log'),'network_policy':'offline',
            }
            out=vllm_impl.plan_start(inp,{'offline':True}); out.pop('shutdown')
            with self.assertRaises(Exception):
                validate_operation_output(self.schemas,'backend','plan_start',out,input_obj=inp)

    def test_process_run_preserves_primary_when_interrupt_cleanup_fails(self):
        from unittest.mock import patch
        pm=ProcessManager(self.schemas)
        spec={"schema_version":"1.0","argv":[sys.executable,"-V"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
        class FakeProcess:
            def communicate(self, timeout=None):
                raise ValueError('primary-failure')
        fake=SimpleNamespace(process=FakeProcess(),pid=123)
        with patch.object(pm,'start',return_value=fake), patch.object(pm,'stop',side_effect=ProcessError('cleanup-failure')):
            with self.assertRaisesRegex(ValueError,'primary-failure') as ctx:
                pm.run(spec)
        self.assertIsInstance(getattr(ctx.exception,'_model_eval_cleanup_error',None),ProcessError)

    def test_process_timeout_preserves_timeout_when_cleanup_fails(self):
        from unittest.mock import patch
        pm=ProcessManager(self.schemas)
        spec={"schema_version":"1.0","argv":[sys.executable,"-V"],"timeout_seconds":0.01,"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
        class FakeProcess:
            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd='fake',timeout=timeout)
        fake=SimpleNamespace(process=FakeProcess(),pid=123)
        with patch.object(pm,'start',return_value=fake), patch.object(pm,'stop',side_effect=ProcessError('cleanup-failure')):
            with self.assertRaisesRegex(ProcessError,'process timed out') as ctx:
                pm.run(spec)
        self.assertIsInstance(getattr(ctx.exception,'_model_eval_cleanup_error',None),ProcessError)
        self.assertIn('cleanup-failure',str(getattr(ctx.exception,'_model_eval_cleanup_error')))

    def test_matrix_cleanup_incomplete_is_always_hard_stop(self):
        exc=ProcessError('backend OOM',details={'cleanup_status':'incomplete'})
        self.assertTrue(MatrixExecutor._failure_requires_hard_stop(exc,'incomplete'))
        self.assertFalse(MatrixExecutor._failure_requires_hard_stop(ProcessError('ordinary failure'),'clean'))

    def test_process_start_ownership_write_failure_does_not_leave_child(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            spec={"schema_version":"1.0","argv":[sys.executable,"-c","import time; time.sleep(30)"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            started=[]
            real_popen=subprocess.Popen
            def capture(*args,**kwargs):
                proc=real_popen(*args,**kwargs); started.append(proc); return proc
            with patch('model_evaluation.core.process.manager.subprocess.Popen',side_effect=capture), patch.object(pm,'_write_ownership',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(ProcessError,'process start was not committed'):
                    pm.start(spec)
            self.assertEqual(len(started),1)
            deadline=time.time()+3
            while started[0].poll() is None and time.time()<deadline:
                time.sleep(0.05)
            self.assertIsNotNone(started[0].poll())

    def test_status_best_effort_never_changes_run_outcome(self):
        from unittest.mock import patch
        orch=object.__new__(Orchestrator)
        with tempfile.TemporaryDirectory() as td, \
             patch.object(orch,'_status',side_effect=OSError('disk full')), \
             patch.object(orch,'_append_core_error') as append:
            orch._status_best_effort(Path(td),'SUCCEEDED')
            append.assert_called_once()

    def test_process_start_commit_failure_reports_incomplete_cleanup(self):
        from unittest.mock import patch
        pm=ProcessManager(self.schemas)
        spec={"schema_version":"1.0","argv":[sys.executable,"-V"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
        fake_proc=SimpleNamespace(pid=43210)
        fake_proc.poll=lambda: None
        with patch('model_evaluation.core.process.manager.subprocess.Popen',return_value=fake_proc), \
             patch('model_evaluation.core.process.manager._proc_start_ticks',return_value=123), \
             patch('model_evaluation.core.process.manager._proc_pgid',return_value=43210), \
             patch.object(pm,'_write_ownership',side_effect=OSError('disk full')), \
             patch.object(pm,'_abort_uncommitted',return_value={'status':'incomplete','pid':43210,'pgid':43210}):
            with self.assertRaisesRegex(ProcessError,'process start was not committed') as ctx:
                pm.start(spec)
        self.assertEqual(ctx.exception.details.get('cleanup_status'),'incomplete')
        self.assertEqual(getattr(ctx.exception,'_model_eval_cleanup_status',None),'incomplete')

    def test_process_run_propagates_incomplete_cleanup_to_primary(self):
        from unittest.mock import patch
        pm=ProcessManager(self.schemas)
        spec={"schema_version":"1.0","argv":[sys.executable,"-V"],"timeout_seconds":0.01,"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
        class FakeProcess:
            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd='fake',timeout=timeout)
        fake=SimpleNamespace(process=FakeProcess(),pid=123)
        cleanup=ProcessError('cleanup incomplete',details={'cleanup_status':'incomplete'})
        with patch.object(pm,'start',return_value=fake), patch.object(pm,'stop',side_effect=cleanup):
            with self.assertRaisesRegex(ProcessError,'process timed out') as ctx:
                pm.run(spec)
        self.assertEqual(ctx.exception.details.get('cleanup_status'),'incomplete')
        self.assertEqual(getattr(ctx.exception,'_model_eval_cleanup_status',None),'incomplete')

    def test_orchestration_signal_guard_ignores_repeat_signal_after_first_interrupt(self):
        import signal as _signal
        from model_evaluation.core.process.signals import orchestration_signal_guard
        from model_evaluation.core.errors import OrchestrationInterruptedError
        with orchestration_signal_guard():
            with self.assertRaises(OrchestrationInterruptedError):
                _signal.raise_signal(_signal.SIGTERM)
            # Cleanup/finalization under the same guard must not be interrupted
            # by a repeated Ctrl+C/SIGTERM.
            _signal.raise_signal(_signal.SIGTERM)

    def test_managed_backend_attach_port_must_match_planned_endpoint(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); model=root/'model'; model.mkdir()
            inp={
                'model':{'id':'m'},
                'deployment':{'management':{'mode':'managed'},'model_location':{'local_path':str(model)},'parameters':{'port':8091}},
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},'log_path':str(root/'backend.log'),'network_policy':'offline',
            }
            out=vllm_impl.plan_start(inp,{'offline':True})
            out['attach']['base_url']='http://127.0.0.1:9999/v1'
            with self.assertRaisesRegex(Exception,'planned endpoint port'):
                validate_operation_output(self.schemas,'backend','plan_start',out,input_obj=inp)

    def test_external_plan_records_reproducibility_inputs_without_trust_bundle(self):
        app=Application(PACKAGE_ROOT, ROOT); plan=app.plan('external_mmlu_example')
        self.assertTrue(plan['plan_id'].startswith('plan-'))
        self.assertIn('model_source',plan['resolved'])
        self.assertNotIn('integrity',plan['resolved'])
        self.assertTrue(all(set(row)=={'kind','name','version'} for row in plan['adapters']))
        self.assertFalse(any(row['adapter']=='core/model-source' for row in plan['warnings']))

    def test_load_plan_checks_stable_plan_identity_for_correlation(self):
        app=Application(PACKAGE_ROOT, ROOT); plan=app.plan('external_mmlu_example')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'plan.json'; path.write_text(json.dumps(plan),encoding='utf-8')
            self.assertEqual(app.load_plan(path)['plan_id'],plan['plan_id'])
            plan['resolved']['dataset_resolution']['revision']='different-recorded-revision'
            path.write_text(json.dumps(plan),encoding='utf-8')
            with self.assertRaisesRegex(ConfigError,'plan_id does not match'):
                app.load_plan(path)

    def test_deployment_template_resolves_model_and_platform(self):
        dep={"schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},"model_location":{"root_template":"{platform.metadata.model_root}","path_template":"{model.source.ref}"}}
        model={"schema_version":"1.0","id":"m","source":{"type":"registry","ref":"Org/Model"}}
        platform={"schema_version":"1.1","id":"p","device":{"adapter":"cpu"},"runtime":{"adapter":"cpu"},"backend_environment":{"provider":"current","profile":"current"},"evaluation_environment":{"provider":"current","profile":"current"},"metadata":{"model_root":"/models"}}
        effective,res=resolve_deployment_profile(dep,model,platform)
        self.assertEqual(effective['model_location']['local_path'],'/models/Org/Model')
        self.assertEqual(res['mode'],'template')

    def test_deployment_model_root_is_canonicalized_across_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); canonical=base/'data'; canonical.mkdir(); alias=base/'aoss'; alias.symlink_to(canonical, target_is_directory=True)
            (canonical/'Org'/'Model').mkdir(parents=True)
            dep={"schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},"model_location":{"root":str(alias),"path_template":"{model.source.ref}"}}
            model={"id":"m","source":{"type":"local","ref":"Org/Model"}}
            effective,_=resolve_deployment_profile(dep,model,{})
            self.assertEqual(effective['model_location']['local_path'],str((canonical/'Org'/'Model').resolve()))

    def test_deployment_resolves_logical_model_tokenizer_under_machine_root(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'models'; root.mkdir()
            dep={"schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},"model_location":{"root":str(root),"path_template":"{model.source.ref}"}}
            model={"id":"m","source":{"type":"hf","ref":"Jackrong/Negentropy-claude-opus-4.7-4B"},"tokenizer":{"ref":"coder3101/Qwen3.5-4B-heretic"}}
            effective,res=resolve_deployment_profile(dep,model,{})
            expected=str((root/'coder3101'/'Qwen3.5-4B-heretic').resolve())
            self.assertEqual(effective['model_location']['tokenizer_path'],expected)
            self.assertEqual(res['tokenizer_ref'],'coder3101/Qwen3.5-4B-heretic')
            self.assertEqual(res['resolved_tokenizer_path'],expected)

    def test_deployment_rejects_logical_model_tokenizer_root_escape(self):
        dep={"schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},"model_location":{"root":"/models","path_template":"{model.source.ref}"}}
        model={"id":"m","source":{"type":"hf","ref":"Org/Model"},"tokenizer":{"ref":"../escape"}}
        with self.assertRaisesRegex(ConfigError,'model.tokenizer.ref escapes configured root'):
            resolve_deployment_profile(dep,model,{})

    def test_vllm_consumes_only_resolved_absolute_model_tokenizer_path(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'models'
            model_dir=root/'Jackrong'/'Negentropy-claude-opus-4.7-4B'
            tokenizer_dir=root/'coder3101'/'Qwen3.5-4B-heretic'
            model_dir.mkdir(parents=True); tokenizer_dir.mkdir(parents=True)
            source_dep={
                'management':{'mode':'managed'},
                'model_location':{'root':str(root),'path_template':'{model.source.ref}'},
                'parameters':{'port':18092},
            }
            model={
                'id':'negentropy-4b',
                'source':{'type':'hf','ref':'Jackrong/Negentropy-claude-opus-4.7-4B'},
                'tokenizer':{'ref':'coder3101/Qwen3.5-4B-heretic'},
            }
            deployment,_=resolve_deployment_profile(source_dep,model,{})
            expected=str(tokenizer_dir.resolve())
            inputs={
                'model':model,
                'deployment':deployment,
                'endpoint':{'host':'127.0.0.1','port':18092},
                'network_policy':'offline',
            }

            vllm_impl.requirements(inputs,{})
            start=vllm_impl.plan_start(inputs,{'offline':True})
            tokenizer_index=start['process']['argv'].index('--tokenizer')+1
            self.assertEqual(start['process']['argv'][tokenizer_index],expected)
            self.assertEqual(start['attach']['tokenizer_path'],expected)

            preflight=vllm_impl.plan_preflight(inputs,{'offline':True})
            probe=next(row for row in preflight['probes'] if row['id']=='model.config')
            payload_index=probe['process']['argv'].index('--payload')+1
            self.assertEqual(json.loads(probe['process']['argv'][payload_index])['tokenizer'],expected)

    def test_deployment_override_only_changes_non_structural_fields(self):
        dep={"schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},"model_location":{"root":"/models","path_template":"{model.source.ref}"},"parameters":{"tensor_parallel_size":1}}
        model={"id":"m","source":{"type":"registry","ref":"Org/Model"}}
        effective,res=resolve_deployment_profile(dep,model,{}, {"parameters":{"tensor_parallel_size":2}})
        self.assertEqual(effective['parameters']['tensor_parallel_size'],2)
        with self.assertRaises(ConfigError): resolve_deployment_profile(dep,model,{}, {"backend":{"adapter":"other"}})

    def test_deployment_template_rejects_root_escape(self):
        dep={"schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},"model_location":{"root":"/models","path_template":"{model.source.ref}"}}
        model={"id":"m","source":{"type":"registry","ref":"../escape"}}
        with self.assertRaises(ConfigError): resolve_deployment_profile(dep,model,{})

    def test_matrix_plan_identifier_changes_with_normalized_content(self):
        obj={"schema_version":"1.0","matrix_id":"matrix-pending","matrix_spec":{"schema_version":"1.0","id":"x","models":["m"],"platforms":["p"],"deployments":["d"],"benchmarks":["b"],"evaluations":["e"]},"plans":[],"summary":{"runs":0}}
        finalize_matrix_plan(obj); original=obj['matrix_id']; obj['summary']['runs']=1; finalize_matrix_plan(obj); self.assertNotEqual(original,obj['matrix_id'])

    def test_execution_environment_drops_unrelated_secrets(self):
        env=execution_subprocess_env({'PATH':'/bin','HOME':'/tmp','MODEL_API_KEY':'secret','HF_TOKEN':'secret2','CUDA_HOME':'/opt/cuda'})
        self.assertEqual(env['PATH'],'/bin'); self.assertNotIn('CUDA_HOME',env); self.assertNotIn('MODEL_API_KEY',env); self.assertNotIn('HF_TOKEN',env)

    def test_unknown_run_override_rejected(self):
        with self.assertRaises(ConfigError): validate_run_overrides({'overrides':{'offine':True}})
        validate_run_overrides({'overrides':{'offline':True,'dataset_timeout_seconds':2,'deployment':{'parameters':{'x':1}}}})

    def test_result_relocation_resolves_only_declared_old_root(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve(); current=base/'current'; current.mkdir(); old=base/'old'
            (current/'RELOCATION.json').write_text(json.dumps({
                'schema_version':'1.0',
                'mappings':[{'old_root':str(old),'new_root':str(current)}],
            }))
            relocation=load_result_relocation(current)
            self.assertEqual(relocation.relocate(str(old/'run'/'raw.json'),label='raw'),current/'run'/'raw.json')
            outside=base/'old-suffix'/'run'
            self.assertEqual(relocation.relocate(str(outside),label='outside'),outside)

    def test_result_relocation_rejects_unsafe_maps(self):
        cases=(
            lambda base,current:{'schema_version':'1.0','mappings':[{'old_root':str(base/'old'),'new_root':str(base/'wrong')}]},
            lambda base,current:{'schema_version':'1.0','mappings':[{'old_root':str(current),'new_root':str(current)}]},
            lambda base,current:{'schema_version':'1.0','mappings':[
                {'old_root':str(base/'old'),'new_root':str(current)},
                {'old_root':str(base/'old'/'nested'),'new_root':str(current)},
            ]},
            lambda base,current:{'schema_version':'1.0','mappings':[{'old_root':'/','new_root':str(current)}]},
        )
        for make in cases:
            with self.subTest(case=make):
                with tempfile.TemporaryDirectory() as td:
                    base=Path(td).resolve(); current=base/'current'; current.mkdir()
                    (current/'RELOCATION.json').write_text(json.dumps(make(base,current)))
                    with self.assertRaises(ConfigError): load_result_relocation(current)
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve(); current=base/'current'; current.mkdir()
            target=base/'map.json'; target.write_text('{}'); (current/'RELOCATION.json').symlink_to(target)
            with self.assertRaises(ConfigError): load_result_relocation(current)

    def test_matrix_success_validation_supports_byte_preserving_relocation(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve(); old=base/'old-results'; current=base/'current-results'; current.mkdir()
            run_id='run-relocated'; run_dir=current/run_id; (run_dir/'config').mkdir(parents=True); (run_dir/'framework_output').mkdir()
            raw=run_dir/'framework_output'/'raw.json'; raw.write_text('{"ok":true}\n')
            old_run=old/run_id
            plan={
                'plan_id':'plan-relocated',
                'run_spec':{'model':'model-x','benchmark':'benchmark-x'},
                'resolved':{'specs':{'evaluation':{'framework':{'adapter':'lm_eval'}}}},
            }
            (run_dir/'config'/'execution_plan.json').write_text(json.dumps({'plan_id':'plan-relocated'}))
            (run_dir/'terminal_record.json').write_text(json.dumps({'outcome':'success'}))
            (run_dir/'canonical_result.json').write_text(json.dumps({
                'schema_version':'1.0','run_id':run_id,'model':'model-x','benchmark':'benchmark-x','framework':'lm_eval',
                'metrics':{},'raw_result':{'path':str(old_run/'framework_output'/'raw.json'),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()},
            }))
            before={p.relative_to(run_dir).as_posix():p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
            (current/'RELOCATION.json').write_text(json.dumps({
                'schema_version':'1.0','mappings':[{'old_root':str(old),'new_root':str(current)}],
            }))
            executor=object.__new__(MatrixExecutor); executor.results_root=current; executor.app=SimpleNamespace(schemas=self.schemas)
            executor.result_relocation=load_result_relocation(current)
            rec={'run_dir':str(old_run),'canonical_result_path':str(old_run/'canonical_result.json')}
            ok,reason,result=executor._validate_success_record(rec,plan)
            after={p.relative_to(run_dir).as_posix():p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
        self.assertTrue(ok,reason); self.assertEqual(result['run_id'],run_id); self.assertEqual(before,after)

    def test_result_relocation_does_not_bypass_confinement_or_symlink_checks(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve(); old=base/'old'; current=base/'current'; current.mkdir()
            (current/'RELOCATION.json').write_text(json.dumps({
                'schema_version':'1.0','mappings':[{'old_root':str(old),'new_root':str(current)}],
            }))
            executor=object.__new__(MatrixExecutor); executor.results_root=current; executor.result_relocation=load_result_relocation(current)
            outside=base/'old-suffix'/'run'; outside.mkdir(parents=True)
            with self.assertRaises(ValueError): executor._stored_path(outside,current,label='run',require_dir=True)
            real=current/'real'; real.mkdir(); link=current/'link'; link.symlink_to(real,target_is_directory=True)
            with self.assertRaises(ValueError): executor._stored_path(old/'link',current,label='run',require_dir=True)


    def test_matrix_max_combinations_rejected_before_iteration(self):
        app=SimpleNamespace(matrix_schemas=MatrixSchemas(PACKAGE_ROOT/'schemas'/'user'))
        spec={"schema_version":"1.0","id":"m","models":["a","b"],"platforms":["p1","p2"],"deployments":["d1","d2"],"benchmarks":["b1","b2"],"evaluations":["e1","e2"],"execution":{"max_combinations":31}}
        with self.assertRaises(ConfigError): MatrixPlanner(app).expand(spec)


    def test_evaluation_middleware_parameter_templates_are_rejected(self):
        evaluation={"schema_version":"1.0","id":"e","framework":{"adapter":"x"},"parameters":{"chat_template":"native-template"},"metadata":{"middleware":{"parameter_templates":{"tool_root":"{platform.metadata.evaluation_tools.x.root}"}}}}
        platform={"metadata":{"evaluation_tools":{"x":{"root":"/opt/x"}}}}
        with self.assertRaises(ConfigError):
            resolve_evaluation_profile(evaluation,platform)

    def test_platform_defaults_do_not_inject_implementation_environment(self):
        platform={"schema_version":"1.1","id":"p","evaluation_environment":{"provider":"current","profile":"current"}}
        effective=_apply_spec_defaults("platform",platform)
        self.assertEqual(effective,platform)
        self.assertNotIn("device",effective)
        self.assertNotIn("runtime",effective)
        self.assertNotIn("backend_environment",effective)
        self.schemas.validate("platform_profile",effective)

    def test_platform_requires_explicit_evaluation_environment(self):
        platform={"schema_version":"1.1","id":"p","device":{"adapter":"cpu"},"runtime":{"adapter":"cpu"},"backend_environment":{"provider":"current","profile":"current"}}
        effective=_apply_spec_defaults("platform",platform)
        self.assertEqual(effective,platform)
        with self.assertRaises(Exception):
            self.schemas.validate("platform_profile",effective)

    def test_platform_local_components_are_all_or_none(self):
        valid={"schema_version":"1.1","id":"p","device":{"adapter":"cpu"},"runtime":{"adapter":"cpu"},"backend_environment":{"provider":"conda","profile":"vllm_env"},"evaluation_environment":{"provider":"current","profile":"current"}}
        self.schemas.validate("platform_profile",valid)
        invalid={"schema_version":"1.1","id":"p","runtime":{"adapter":"cpu"},"evaluation_environment":{"provider":"current","profile":"current"}}
        with self.assertRaises(Exception):
            self.schemas.validate("platform_profile",invalid)

    def test_platform_adapter_parameters_are_direct_and_component_scoped(self):
        platform={"runtime":{"adapter":"future","parameters":{"root":"/opt/future","vendor_key":{"x":1}}},"device":{"adapter":"future","parameters":{"tool":"/bin/future"}}}
        self.assertEqual(adapter_parameters(platform,'runtime'),{"root":"/opt/future","vendor_key":{"x":1}})
        self.assertEqual(adapter_parameters(platform,'device'),{"tool":"/bin/future"})
        self.assertEqual(adapter_parameters(platform,'missing'),{})

    def test_matrix_executor_uses_application_host_runtime_namespace(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); app=SimpleNamespace(root=base/'project',host_runtime_root=base/'host-runtime')
            ex=MatrixExecutor(app)
            self.assertEqual(ex.resources.runtime_root,(base/'host-runtime'/'resources').resolve())

    def test_schema_uri_format_is_enforced(self):
        obj={"schema_version":"1.0","service_type":"llm","ownership":"external","model":{"id":"m"},"protocols":{"openai_completion":{"url":"not a uri"}},"capabilities":{"schema_version":"1.0","values":{}},"auth":{"mode":"none"}}
        for invalid in ('not a uri','relative/path','http://','http://example.invalid:bad','éxample:value'):
            obj['protocols']['openai_completion']['url']=invalid
            with self.assertRaises(Exception): self.schemas.validate('service_descriptor',obj)
        for valid in ('http://127.0.0.1:8000/v1','https://example.invalid/v1','http://[::1]:8000/v1','urn:model-eval:transport'):
            obj['protocols']['openai_completion']['url']=valid
            self.schemas.validate('service_descriptor',obj)

    def test_backend_attach_semantics_are_core_enforced(self):
        out={"schema_version":"1.0","attach":{"base_url":"http://user:pass@example.com/v1","model_id":"m","ownership":"external","auth":{"mode":"none"}},"readiness":{"timeout_seconds":1}}
        inp={"deployment":{"management":{"mode":"external"}}}
        with self.assertRaises(Exception): validate_operation_output(self.schemas,'backend','plan_start',out,input_obj=inp)
        out['attach']['base_url']='https://example.com/v1'; out['attach']['ownership']='attached'
        with self.assertRaises(Exception): validate_operation_output(self.schemas,'backend','plan_start',out,input_obj=inp)

    def test_evaluator_normalize_does_not_require_raw_sha(self):
        out={"schema_version":"1.0","run_id":"r","model":"m","benchmark":"b","framework":"f","metrics":{},"raw_result":{"path":"/tmp/result.json"}}
        validate_operation_output(self.schemas,'evaluator','normalize',out,input_obj={})

    def test_strict_task_artifact_must_be_core_confined(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); stage=root/'task'; stage.mkdir(); outside=root/'outside.yaml'; outside.write_text('x')
            import hashlib
            task={"provenance":{"strict":True},"task_root":str(stage),"artifacts":[{"path":str(outside),"sha256":hashlib.sha256(b'x').hexdigest()}]}
            with self.assertRaises(CompatibilityError): Orchestrator._verify_task_artifacts(task,stage)

    def test_strict_task_rejects_in_tree_symlink_before_resolve(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); stage=root/'task'; stage.mkdir(); real=stage/'real.yaml'; real.write_text('x'); link=stage/'link.yaml'; link.symlink_to(real)
            import hashlib
            task={"provenance":{"strict":True},"task_root":str(stage),"artifacts":[{"path":str(link),"sha256":hashlib.sha256(b'x').hexdigest()}]}
            with self.assertRaises(CompatibilityError): Orchestrator._verify_task_artifacts(task,stage)

    def test_canonical_raw_result_rejects_in_tree_symlink_before_resolve(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); real=root/'real.json'; real.write_text('{}'); link=root/'link.json'; link.symlink_to(real)
            result={"raw_result":{"path":str(link)}}
            with self.assertRaises(CompatibilityError): Orchestrator._verify_canonical_raw_result(result,root)

    def test_process_start_stop_persists_and_clears_ownership(self):
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            spec={"schema_version":"1.0","argv":[sys.executable,"-c","import time; time.sleep(5)"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            h=pm.start(spec); self.assertTrue(list((Path(td)/'proc').glob('process-*.json'))); pm.stop(h,grace_seconds=0.05,kill_seconds=1.0); self.assertFalse(list((Path(td)/'proc').glob('process-*.json')))


    def test_process_graceful_shutdown_precedes_force_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            code=(
                "import signal,time,sys; "
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
                "print('ready', flush=True); time.sleep(30)"
            )
            spec={"schema_version":"1.0","argv":[sys.executable,"-c",code],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            h=pm.start(spec); self.assertEqual(h.process.stdout.readline().decode().strip(),'ready')
            report=pm.stop_with_report(h,grace_seconds=1.0,kill_seconds=0.5)
            self.assertEqual(report['status'],'clean')
            self.assertEqual(report['graceful']['result'],'success')
            self.assertFalse(report['fallback']['sigkill_attempted'])
            again=pm.stop_with_report(h,grace_seconds=0.01,kill_seconds=0.01)
            self.assertEqual(again['status'],'clean')
            self.assertEqual(again['graceful']['result'],'already_exited')

    def test_process_force_fallback_is_bounded_and_owned(self):
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            code="import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)"
            spec={"schema_version":"1.0","argv":[sys.executable,"-c",code],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            h=pm.start(spec); self.assertEqual(h.process.stdout.readline().decode().strip(),'ready')
            report=pm.stop_with_report(h,grace_seconds=0.05,kill_seconds=1.0)
            self.assertEqual(report['status'],'clean')
            self.assertEqual(report['graceful']['result'],'timeout')
            self.assertTrue(report['fallback']['sigkill_attempted'])
            self.assertEqual(report['fallback']['result'],'success')
            self.assertFalse(report['owned_process_group_remaining'])
            self.assertFalse(list((Path(td)/'proc').glob('process-*.json')))

    def test_process_graceful_signal_targets_leader_before_group_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); marker=root/'child-got-term'
            pm=ProcessManager(self.schemas,ownership_root=root/'proc')
            child_code=(
                "import signal,time,pathlib; "
                f"p=pathlib.Path({str(marker)!r}); "
                "signal.signal(signal.SIGTERM, lambda *_: p.write_text('term')); "
                "print('child-ready', flush=True); time.sleep(30)"
            )
            parent_code=(
                "import signal,subprocess,sys,time; "
                f"c=subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.PIPE,text=True); "
                "assert c.stdout.readline().strip() == 'child-ready'; "
                "print('ready', flush=True); "
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(30)"
            )
            spec={"schema_version":"1.0","argv":[sys.executable,"-c",parent_code],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            h=pm.start(spec); self.assertEqual(h.process.stdout.readline().decode().strip(),'ready')
            # Give the child enough time to install its handler and join the same PGID.
            deadline=time.time()+2
            while time.time()<deadline:
                members=__import__('model_evaluation.core.process.manager',fromlist=['_proc_group_members'])._proc_group_members(h.pgid)
                if len(members)>=2: break
                time.sleep(0.02)
            report=pm.stop_with_report(h,grace_seconds=0.05,kill_seconds=1.0)
            self.assertEqual(report['graceful']['target'],'leader')
            self.assertEqual(report['fallback']['target'],'process_group')
            self.assertTrue(report['fallback']['sigkill_attempted'])
            self.assertFalse(marker.exists(), 'child received graceful SIGTERM; graceful phase must target leader only')
            self.assertEqual(report['status'],'clean')

    def test_process_cleanup_housekeeping_errors_are_contained(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            code="import signal,time,sys; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); print('ready', flush=True); time.sleep(30)"
            spec={"schema_version":"1.0","argv":[sys.executable,"-c",code],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            h=pm.start(spec); self.assertEqual(h.process.stdout.readline().decode().strip(),'ready')
            with patch.object(pm,'_remove_ownership',side_effect=OSError('unlink failed')), patch.object(pm,'_close',side_effect=OSError('close failed')):
                report=pm.stop_with_report(h,grace_seconds=1.0,kill_seconds=0.5)
            self.assertEqual(report['status'],'clean')
            phases={x['phase'] for x in report.get('secondary_errors',[])}
            self.assertEqual(phases,{'remove_ownership','close_handles'})
            # Close the inherited capture handles after the injected housekeeping failure.
            pm._close(h)

    def test_process_cleanup_does_not_signal_ambiguous_orphan_group(self):
        from unittest.mock import patch
        pm=ProcessManager(self.schemas)
        class GoneProcess:
            def poll(self): return 0
        fake=SimpleNamespace(process=GoneProcess(),pid=43210,pgid=43210,start_ticks=123,ownership_path=None,stdout_handle=None,stderr_handle=None)
        with patch.object(ProcessManager,'_group_alive',return_value=True), patch('model_evaluation.core.process.manager._proc_start_ticks',return_value=None), patch('model_evaluation.core.process.manager.os.killpg') as killpg:
            report=pm.stop_with_report(fake,grace_seconds=0.01,kill_seconds=0.01)
        self.assertEqual(report['status'],'incomplete')
        self.assertEqual(report['graceful']['result'],'ownership_ambiguous')
        killpg.assert_not_called()

    def test_failure_record_preserves_structured_error_and_redacts_secret(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); run=root/'run-x'; (run/'logs').mkdir(parents=True)
            secret='super-secret-token-value'
            (run/'logs'/'backend.log').write_text(f'backend failed token={secret}\nCUDA out of memory\n',encoding='utf-8')
            pm=ProcessManager(self.schemas,secrets=SecretStore({'secret://x':secret}),ownership_root=root/'proc')
            orch=Orchestrator(project_root=PACKAGE_ROOT,schemas=self.schemas,registry=None,process_manager=pm,resource_manager=None,results_root=root/'results',cache_root=root/'cache')
            exc=AdapterExecutionError('BACKEND_FAILED',f'backend rejected {secret}',retryable=False,details={'why':secret})
            record=orch._failure_record(run,stage='SERVICE_STARTING',failure=exc,cleanup={'schema_version':'1.0','backend':{'status':'clean'}})
            self.assertEqual(record['primary_error']['code'],'BACKEND_FAILED')
            self.assertEqual(record['stage'],'SERVICE_STARTING')
            rendered=json_dumps_strict(record)
            self.assertNotIn(secret,rendered)
            self.assertIn('<redacted>',rendered)
            self.assertIn('CUDA out of memory',rendered)

    def test_failure_log_tail_is_bounded_and_does_not_use_read_text(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); log=root/'backend.log'
            with log.open('wb') as f:
                f.seek(2*1024*1024)
                f.write(b'prefix\nfinal-one\nfinal-two\n')
            pm=ProcessManager(self.schemas)
            orch=Orchestrator(project_root=PACKAGE_ROOT,schemas=self.schemas,registry=None,process_manager=pm,resource_manager=None,results_root=root/'results',cache_root=root/'cache')
            with patch.object(Path,'read_text',side_effect=AssertionError('unbounded read_text forbidden')):
                tail=orch._log_tail(log,max_lines=2,max_chars=100,max_bytes=4096)
            self.assertEqual(tail,['final-one','final-two'])

    def test_runtime_version_record_is_redacted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); run=root/'run'; (run/'config').mkdir(parents=True)
            secret='runtime-version-secret-value'
            pm=ProcessManager(self.schemas,secrets=SecretStore({'secret://x':secret}))
            orch=Orchestrator(project_root=PACKAGE_ROOT,schemas=self.schemas,registry=None,process_manager=pm,resource_manager=None,results_root=root/'results',cache_root=root/'cache')
            orch._save_runtime_versions(run,{'schema_version':'1.0','warnings':[f'probe failed: {secret}']})
            obj=json_loads_strict((run/'config'/'runtime_versions.json').read_text())
            self.assertEqual(obj['schema_version'],'1.0')
            self.assertNotIn(secret,json_dumps_strict(obj))

    def test_capability_namespace_is_owned_by_producer(self):
        desc={"provider":"current","identity":"x","capabilities":{"values":{"service.generation":True}}}
        with self.assertRaises(CompatibilityError): facts_from_environment(desc,'evaluation_environment')

    def test_persisted_secret_guard_is_broad_but_allows_semantics(self):
        with self.assertRaises(ConfigError): _reject_inline_secrets({'vendor_api_key':'clear'})
        with self.assertRaises(ConfigError): _reject_inline_secrets({'client_secret':'clear'})
        _reject_inline_secrets({'auth_mode':'bearer','vendor_api_key':'secret://env/VENDOR_API_KEY'})

    def test_strict_json_rejects_nonfinite_and_duplicate_keys(self):
        with self.assertRaises(ValueError): json_dumps_strict({'x':float('nan')})
        with self.assertRaises(ValueError): json_loads_strict('{"x":NaN}')
        with self.assertRaises(ValueError): json_loads_strict('{"x":1,"x":2}')

    def test_yaml_duplicate_key_rejected(self):
        with self.assertRaises(ValueError): load_yaml_strict('a: 1\na: 2\n')

    def test_manifest_rejects_legacy_hidden_user_config_protocol(self):
        manifest={
            "adapter_api":"1.0","kind":"backend","name":"x","version":"1.0.0","operations":["requirements"],
            "implementation":{"language":"python","user_config":{}},
        }
        with self.assertRaises(Exception):
            self.schemas.validate('adapter_manifest',manifest)

    def test_platform_schema_rejects_legacy_functional_metadata_path(self):
        platform={
            "schema_version":"1.1","id":"p","evaluation_environment":{"provider":"current","profile":"current"},
            "metadata":{"middleware":{"adapter_parameters":{"runtime":{"root":"/x"}}}},
        }
        with self.assertRaises(Exception):
            self.schemas.validate('platform_profile',platform)

    def test_deployment_schema_rejects_runtime_compatibility_in_parameters(self):
        deployment={
            "schema_version":"1.1","id":"d","backend":{"adapter":"vllm"},"management":{"mode":"managed"},
            "compatibility":{"runtime_families":["cuda"]},"parameters":{"runtime_families":["cuda"]},
        }
        with self.assertRaises(Exception):
            self.schemas.validate('deployment_profile',deployment)

    def test_adapter_schema_versions_and_name_are_enforced(self):
        _validate_schema_versions({'schema_versions':{'service_descriptor':'1.0'}})
        with self.assertRaises(Exception): _validate_schema_versions({'schema_versions':{'service_descriptor':'2.0'}})
        with self.assertRaises(Exception): _validate_schema_versions({'schema_versions':{'future_unknown':'1.0'}})
        with self.assertRaises(Exception): _validate_adapter_name('vendor/name')

    def test_environment_adapter_parameters_are_direct(self):
        platform={"backend_environment":{"provider":"future","profile":"x","parameters":{"image":"x"}},"evaluation_environment":{"provider":"future","profile":"y","parameters":{"venv":"/x"}}}
        self.assertEqual(adapter_parameters(platform,'backend_environment'),{'image':'x'})
        self.assertEqual(adapter_parameters(platform,'evaluation_environment'),{'venv':'/x'})

    def test_process_run_cleans_descendant_group_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            code="import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); sys.exit(0)"
            spec={"schema_version":"1.0","argv":[sys.executable,"-c",code],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},"timeout_seconds":2}
            with self.assertRaises(ProcessError): pm.run(spec)
            self.assertFalse(list((Path(td)/'proc').glob('process-*.json')))

    def test_process_run_allows_transient_group_disappearance_after_leader_exit(self):
        """A killpg/procfs race must not turn a completed probe into failure."""
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            spec={"schema_version":"1.0","argv":[sys.executable,"-c","print('ok')"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},"timeout_seconds":2}
            observations=iter((True,False))
            with patch.object(ProcessManager,'_group_alive',side_effect=lambda _pgid: next(observations)):
                result=pm.run(spec)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout.strip(),b'ok')
            self.assertFalse(list((Path(td)/'proc').glob('process-*.json')))

    def test_stale_recovery_keeps_ambiguous_orphan_group_record(self):
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'proc'; root.mkdir()
            record=root/'process-123-x.json'; record.write_text(json.dumps({'pid':123,'pgid':123,'start_ticks':9,'boot_id':'same-boot'}))
            pm=ProcessManager(self.schemas,ownership_root=root)
            with patch('model_evaluation.core.process.manager._linux_boot_id',return_value='same-boot'), patch('model_evaluation.core.process.manager._proc_start_ticks',return_value=None), patch.object(ProcessManager,'_group_alive',return_value=True):
                rows=pm.recover_stale_managed()
            self.assertEqual(rows[0]['status'],'orphaned_group_ambiguous'); self.assertTrue(record.exists())

    def test_adapter_sdk_json_is_strict(self):
        with self.assertRaises(ValueError): adapter_json_dumps({'x':float('inf')})
        with self.assertRaises(ValueError): adapter_json_loads('{"x":1,"x":2}')

    def test_http_redirect_handler_refuses_redirects(self):
        self.assertIsNone(_NoRedirect().redirect_request(None,None,302,'Found',{},'https://other.example/'))

    def test_matrix_repository_applies_persisted_secret_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.yaml'
            p.write_text("""schema_version: "1.0"
id: m
models: [a]
platforms: [p]
deployments: [d]
benchmarks: [b]
evaluations: [e]
overrides:
  vendor_api_key: cleartext
""")
            repo=MatrixRepository(Path(td),MatrixSchemas(PACKAGE_ROOT/'schemas'/'user'))
            with self.assertRaises(ConfigError): repo.load(p)

    def test_matrix_safe_path_rejects_in_tree_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); real=root/'real.json'; real.write_text('{}'); link=root/'link.json'; link.symlink_to(real)
            with self.assertRaises(ValueError): MatrixExecutor._safe_confined_path(link,root,label='x',require_file=True)

    def test_lm_eval_bearer_always_uses_core_controlled_proxy(self):
        from unittest.mock import patch
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        service={
            'model':{'id':'m'},
            'protocols':{'openai_completion':{'url':'https://trusted.example/v1/completions'}},
            'capabilities':{'values':{}},
            'tokenizer':{'mode':'local','path':'/tmp/tokenizer'},
            'auth':{'mode':'bearer','secret_ref':'secret://env/MODEL_API_KEY'},
        }
        task={'task_id':'x','protocol_fingerprint':'p','execution':{'inference':['generation']}}
        evaluation={'parameters':{'tool_root':'/unused'}}
        with patch.object(lm_eval_impl,'_framework_source',return_value=(Path('/tmp'),'rev',False)):
            out=lm_eval_impl.plan_evaluate({'service':service,'task':task,'evaluation':evaluation,'output_root':'/tmp/out','run_metadata':{}},{})
        proc=out['process']
        self.assertEqual(proc['metadata']['transport_proxy'],'inject')
        self.assertEqual(proc['secret_env']['MODEL_EVAL_UPSTREAM_API_KEY'],'secret://env/MODEL_API_KEY')
        self.assertNotIn('OPENAI_API_KEY',proc.get('secret_env',{}))

    def test_lm_eval_limit_is_explicit_adapter_owned_smoke_parameter(self):
        from unittest.mock import patch
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        service={
            'model':{'id':'m'},'protocols':{'openai_completion':{'url':'http://127.0.0.1:8091/v1/completions'}},
            'capabilities':{'values':{}},'tokenizer':{'mode':'local','path':'/tmp/tokenizer'},
            'auth':{'mode':'none'},
        }
        task={'task_id':'mmlu_abstract_algebra','protocol_fingerprint':'p','execution':{'inference':['multiple_choice'],'num_fewshot':0}}
        evaluation={'parameters':{'tool_root':'/unused','limit':1,'log_samples':False}}
        with patch.object(lm_eval_impl,'_framework_source',return_value=(Path('/tmp'),'rev',False)):
            out=lm_eval_impl.plan_evaluate({'service':service,'task':task,'evaluation':evaluation,'output_root':'/tmp/out','run_metadata':{}},{})
        argv=out['process']['argv']
        self.assertEqual(argv[argv.index('--limit')+1],'1')
        self.assertNotIn('--log_samples',argv)

    def test_lm_eval_samples_are_opt_in_when_parameter_is_omitted(self):
        from unittest.mock import patch
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        service={
            'model':{'id':'m'},'protocols':{'openai_completion':{'url':'http://127.0.0.1:8091/v1/completions'}},
            'capabilities':{'values':{}},'tokenizer':{'mode':'local','path':'/tmp/tokenizer'},
            'auth':{'mode':'none'},
        }
        task={'task_id':'task','protocol_fingerprint':'p','execution':{'inference':['generation']}}
        evaluation={'parameters':{'tool_root':'/unused'}}
        with patch.object(lm_eval_impl,'_framework_source',return_value=(Path('/tmp'),'rev',False)):
            out=lm_eval_impl.plan_evaluate({'service':service,'task':task,'evaluation':evaluation,'output_root':'/tmp/out','run_metadata':{}},{})
        self.assertNotIn('--log_samples',out['process']['argv'])

    def test_lm_eval_plan_evaluate_records_all_reproducibility_seeds(self):
        from unittest.mock import patch
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        service={
            'model':{'id':'m'},'protocols':{'openai_completion':{'url':'http://127.0.0.1:8091/v1/completions'}},
            'capabilities':{'values':{}},'tokenizer':{'mode':'local','path':'/tmp/tokenizer'},
            'auth':{'mode':'none'},
        }
        task={'task_id':'bbh','protocol_fingerprint':'p','execution':{'inference':['multiple_choice']}}
        evaluation={'parameters':{
            'tool_root':'/unused','random_seed':1,'numpy_random_seed':2,
            'torch_random_seed':3,'fewshot_random_seed':4,'request_seed':5,
            'pythonhashseed':6,
        }}
        with patch.object(lm_eval_impl,'_framework_source',return_value=(Path('/tmp'),'rev',False)):
            process=lm_eval_impl.plan_evaluate({'service':service,'task':task,'evaluation':evaluation,'output_root':'/tmp/out','run_metadata':{}},{})['process']
        argv=process['argv']
        self.assertEqual(argv[argv.index('--seed')+1],'1,2,3,4')
        self.assertIn('seed=5',argv[argv.index('--model_args')+1].split(','))
        self.assertEqual(process['metadata']['reproducibility'],{
            'random_seed':1,'numpy_random_seed':2,'torch_random_seed':3,
            'fewshot_random_seed':4,'request_seed':5,'pythonhashseed':6,
        })
        self.assertEqual(process['env_patch']['set']['PYTHONHASHSEED'],'6')

    def test_vllm_plan_start_records_explicit_reproducibility_seeds(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model'; model.mkdir()
            inp={
                'model':{'id':'m'},
                'deployment':{'management':{'mode':'managed'},'model_location':{'local_path':str(model)},'parameters':{'port':8091,'seed':17,'pythonhashseed':23}},
                'platform':{},'endpoint':{'host':'127.0.0.1','port':8091},'log_path':str(Path(td)/'backend.log'),'network_policy':'offline',
            }
            process=vllm_impl.plan_start(inp,{'offline':True})['process']
        self.assertEqual(process['argv'][process['argv'].index('--seed')+1],'17')
        self.assertEqual(process['env_patch']['set']['PYTHONHASHSEED'],'23')

    def test_run_id_is_model_benchmark_short_local_timestamp(self):
        plan={
            'run_spec':{
                'model':'user-model-qwen35-08b-base-deadbeef',
                'platform':'internal-platform',
                'deployment':'internal-deployment',
                'benchmark':'bbh',
            },
            'resolved':{'specs':{
                'model':{'experiment_id':'qwen35-08b-base'},
                'platform':{'device':{'adapter':'mlu'},'metadata':{'result_platform':'mlu'}},
                'deployment':{'backend':{'adapter':'vllm'}},
            }},
        }
        run_id=Orchestrator._run_id(plan)
        self.assertRegex(run_id,r'^mlu_qwen35-08b-base_vllm_bbh_\d{6}-\d{4}$')
        safe=Orchestrator._run_id({'run_spec':{
            'model':'org/model:revision','platform':'hardware/name',
            'deployment':'backend/name','benchmark':'suite/name',
        }})
        self.assertRegex(
            safe,
            r'^hardware-name_org-model-revision_backend-name_suite-name_\d{6}-\d{4}$',
        )
        self.assertNotIn('/',safe)

    def test_bbh_binding_exposes_canonical_accuracy_contract(self):
        import importlib.util
        module_path=PACKAGE_ROOT/'adapters'/'binding'/'lm_eval.bbh'/'impl.py'
        spec=importlib.util.spec_from_file_location('test_lm_eval_bbh_impl',module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        bbh_impl=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bbh_impl)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'harness'; source=root/'lm_eval'/'tasks'/'leaderboard'/'bbh_mc'; source.mkdir(parents=True)
            (source/'_leaderboard_bbh.yaml').write_text(
                'group: leaderboard_bbh\ntask:\n  - leaderboard_bbh_boolean_expressions\n',encoding='utf-8'
            )
            (source/'_fewshot_template_yaml').write_text(
                'dataset_path: SaylorTwift/bbh\noutput_type: multiple_choice\nnum_fewshot: 3\n',encoding='utf-8'
            )
            (source/'boolean_expressions.yaml').write_text(
                'include: _fewshot_template_yaml\ntask: leaderboard_bbh_boolean_expressions\ndataset_name: boolean_expressions\n',encoding='utf-8'
            )
            data=Path(td)/'data'; data.mkdir(); (data/'boolean_expressions.json').write_text('{"examples": []}\n',encoding='utf-8')
            task=bbh_impl.build_task({
                'benchmark':{'id':'bbh','metrics':['accuracy'],'protocol':{'fewshot':7}},
                'dataset_artifact':{'dataset_id':'bbh','root':str(data),'fingerprint':'0'*64},
                'evaluation':{'parameters':{'tool_root':str(root),'provenance_policy':'migration'}},
                'staging_root':str(Path(td)/'task'),
            },{})
        self.assertEqual(task['task_id'],'leaderboard_bbh_local')
        self.assertEqual(task['execution']['num_fewshot'],7)
        self.assertEqual(task['metrics'],{
            'namespace':'canonical','required':['accuracy'],
            'mapping':{'acc_norm,none':'accuracy'},
        })

    def test_bbh_binding_basic_does_not_hash_dataset_bytes(self):
        import importlib.util
        from unittest.mock import patch
        module_path=PACKAGE_ROOT/'adapters'/'binding'/'lm_eval.bbh'/'impl.py'
        spec=importlib.util.spec_from_file_location('test_lm_eval_bbh_basic_impl',module_path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'harness'; source=root/'lm_eval'/'tasks'/'leaderboard'/'bbh_mc'; source.mkdir(parents=True)
            (source/'_leaderboard_bbh.yaml').write_text('group: leaderboard_bbh\ntask:\n  - leaderboard_bbh_boolean_expressions\n',encoding='utf-8')
            (source/'_fewshot_template_yaml').write_text('dataset_path: SaylorTwift/bbh\n',encoding='utf-8')
            (source/'boolean_expressions.yaml').write_text('include: _fewshot_template_yaml\ntask: leaderboard_bbh_boolean_expressions\ndataset_name: boolean_expressions\n',encoding='utf-8')
            data_root=Path(td).resolve()/'data'; data_root.mkdir(); data=data_root/'boolean_expressions.json'; data.write_text('{"examples":[{"input":"q","target":"a"}]}\n',encoding='utf-8')
            real_sha=mod._sha
            def reject_data_hash(path):
                if Path(path).resolve()==data.resolve(): raise AssertionError('binding must not hash basic dataset content')
                return real_sha(path)
            artifact={'dataset_id':'bbh','revision':'declared-rev','root':str(data_root),'files':[{'path':str(data)}],'metadata':{'integrity_policy':'basic','content_fingerprinted':False,'content_verified':False,'revision_provenance':'provider-declared'}}
            with patch.object(mod,'_sha',side_effect=reject_data_hash):
                task=mod.build_task({'benchmark':{'id':'bbh','metrics':['accuracy']},'dataset_artifact':artifact,'evaluation':{'parameters':{'tool_root':str(root),'provenance_policy':'migration'}},'staging_root':str(Path(td)/'task')},{})
            manifest=json.loads((Path(task['task_root'])/'PROTOCOL_MANIFEST.json').read_text(encoding='utf-8'))
        self.assertNotIn('data_sha256',manifest['rows'][0])
        self.assertEqual(manifest['dataset_integrity_policy'],'basic')
        self.assertFalse(task['metadata']['content_fingerprinted'])

    def test_lm_eval_filtered_stderr_is_attached_to_metric(self):
        import json
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            result=Path(td)/'result.json'
            result.write_text(json.dumps({'results':{'task':{'acc,none':0.75,'acc_stderr,none':0.0}},'higher_is_better':{'task':{'acc':True}}}),encoding='utf-8')
            out=lm_eval_impl.normalize({'raw_result_root':td,'task':{'task_id':'task','protocol_fingerprint':'pf'},'run_metadata':{'run_id':'r','model':'m','benchmark':'b'}},{})
        self.assertEqual(out['metrics'],{'acc,none':{'value':0.75,'higher_is_better':True,'stderr':0.0}})

    def test_lm_eval_normalize_ignores_task_metadata_fields(self):
        import json
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            result=Path(td)/'result.json'
            result.write_text(json.dumps({
                'results':{'task':{
                    'name':'task','alias':'Task Alias','sample_len':100,'sample_count':100,
                    'acc,none':0.5,'acc_stderr,none':0.01
                }},
                'higher_is_better':{'task':{'acc':True}}
            }),encoding='utf-8')
            out=lm_eval_impl.normalize({'raw_result_root':td,'task':{'task_id':'task','protocol_fingerprint':'pf'},'run_metadata':{'run_id':'r','model':'m','benchmark':'b'}},{})
        self.assertEqual(out['metrics'],{'acc,none':{'value':0.5,'higher_is_better':True,'stderr':0.01}})
        self.assertEqual(out['metadata']['result_scope'],'task')

    def test_lm_eval_normalize_uses_group_aggregate(self):
        import json
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            result=Path(td)/'result.json'
            result.write_text(json.dumps({
                'results':{
                    'mmlu_anatomy':{'acc,none':0.5,'acc_stderr,none':0.1},
                    'mmlu_astronomy':{'acc,none':1.0,'acc_stderr,none':0.0}
                },
                'groups':{'mmlu':{'alias':'mmlu','acc,none':0.75,'acc_stderr,none':0.05}},
                'group_subtasks':{'mmlu':['mmlu_anatomy','mmlu_astronomy']},
                'higher_is_better':{'mmlu':{'acc':True}}
            }),encoding='utf-8')
            out=lm_eval_impl.normalize({'raw_result_root':td,'task':{'task_id':'mmlu','protocol_fingerprint':'pf'},'run_metadata':{'run_id':'r','model':'m','benchmark':'mmlu'}},{})
        self.assertEqual(out['metrics'],{'acc,none':{'value':0.75,'higher_is_better':True,'stderr':0.05}})
        self.assertEqual(out['metadata']['result_scope'],'group')

    def test_lm_eval_likelihood_requires_echo(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        out=lm_eval_impl.requirements({'task':{'execution':{'inference':['multiple_choice']}}},{})
        paths={x['path'] for x in out['requirements']}
        self.assertIn('service.completion_logprobs',paths)
        self.assertIn('service.echo',paths)
        self.assertIn('service.tokenizer_available',paths)

    def test_model_source_recording_does_not_enforce_or_verify_model_bytes(self):
        bad={'source':{'type':'registry','ref':'Org/Model','revision':'main'},'provenance':{'policy':'pinned'}}
        moving=assess_model_provenance(bad,{'management':{'mode':'external'}})
        self.assertFalse(moving['revision_stable'])
        good={'source':{'type':'registry','ref':'Org/Model','revision':'0123456789abcdef'},'provenance':{'policy':'pinned'}}
        out=assess_model_provenance(good,{'management':{'mode':'external'}})
        self.assertTrue(out['revision_stable']); self.assertEqual(out['recording'],'remote_endpoint')
        local={'source':{'type':'local','ref':'/models/m'},'provenance':{'policy':'pinned'}}
        recorded=assess_model_provenance(local,{'management':{'mode':'managed'},'model_location':{'local_path':'/models/m'}})
        self.assertEqual(recorded['recording'],'local_path')
        self.assertNotIn('artifact_digest',recorded)
        self.assertNotIn('warnings',recorded)

    def test_lm_eval_strict_framework_provenance_requires_revision(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'lm_eval').mkdir()
            with self.assertRaises(AdapterError): lm_eval_impl._framework_source({'tool_root':str(root),'provenance_policy':'strict'})

    def test_doctor_evaluator_rpc_budget_covers_bounded_framework_probes(self):
        from model_evaluation.cli import EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS
        self.assertGreaterEqual(EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS, 30)

    def test_lm_eval_required_cleanliness_fails_closed_when_git_status_is_unknown(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'lm_eval').mkdir()
            rev=Mock(returncode=0,stdout='a'*40+'\n')
            status=Mock(returncode=1,stdout='')
            with patch.object(lm_eval_impl.subprocess,'run',side_effect=[rev,status]):
                with self.assertRaisesRegex(AdapterError,'could not be established'):
                    lm_eval_impl._framework_source({
                        'tool_root':str(root),'provenance_policy':'strict',
                        'framework_revision':'a'*40,'require_clean_framework':True,
                    })

    def test_lm_eval_cleanliness_can_be_disabled_explicitly(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'lm_eval').mkdir()
            rev=Mock(returncode=0,stdout='b'*40+'\n')
            status=Mock(returncode=1,stdout='')
            with patch.object(lm_eval_impl.subprocess,'run',side_effect=[rev,status]):
                _,revision,dirty=lm_eval_impl._framework_source({
                    'tool_root':str(root),'framework_revision':'b'*40,
                    'require_clean_framework':False,
                })
            self.assertEqual(revision,'b'*40)
            self.assertIsNone(dirty)

    def test_lm_eval_revision_falls_back_to_read_only_git_metadata(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'lm_eval').mkdir(); (root/'.git/refs/heads').mkdir(parents=True)
            (root/'.git/HEAD').write_text('ref: refs/heads/main\n',encoding='utf-8')
            (root/'.git/refs/heads/main').write_text('c'*40+'\n',encoding='utf-8')
            with patch.object(lm_eval_impl.subprocess,'run',side_effect=FileNotFoundError('git')):
                _,revision,dirty=lm_eval_impl._framework_source({
                    'tool_root':str(root),'provenance_policy':'strict',
                    'framework_revision':'c'*40,'require_clean_framework':False,
                })
            self.assertEqual(revision,'c'*40)
            self.assertIsNone(dirty)

    def test_git_metadata_reader_supports_worktree_and_packed_refs(self):
        from model_evaluation.sdk.gitmeta import read_git_head
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); worktree=root/'worktree'; worktree.mkdir()
            git_dir=root/'repo.git/worktrees/w'; git_dir.mkdir(parents=True)
            common=root/'repo.git'; (common/'refs').mkdir(exist_ok=True)
            (worktree/'.git').write_text(f'gitdir: {git_dir}\n',encoding='utf-8')
            (git_dir/'commondir').write_text('../..\n',encoding='utf-8')
            (git_dir/'HEAD').write_text('ref: refs/heads/main\n',encoding='utf-8')
            (common/'packed-refs').write_text('# pack-refs with: peeled\n'+'d'*40+' refs/heads/main\n',encoding='utf-8')
            self.assertEqual(read_git_head(worktree),'d'*40)

    def test_git_metadata_reader_rejects_escaping_symbolic_ref(self):
        from model_evaluation.sdk.gitmeta import read_git_head
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'.git').mkdir()
            (root/'.git/HEAD').write_text('ref: refs/heads/../../outside\n',encoding='utf-8')
            self.assertIsNone(read_git_head(root))

    def test_git_metadata_reader_fails_closed_for_malformed_or_oversized_metadata(self):
        from model_evaluation.sdk.gitmeta import read_git_head
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'.git').mkdir()
            for value in (
                'ref: refs/heads/bad\x00name\n',
                'ref: refs//heads/main\n',
                'ref: refs/heads/main.lock\n',
                'ref: refs/heads/topic@{1}\n',
            ):
                (root/'.git/HEAD').write_text(value,encoding='utf-8')
                self.assertIsNone(read_git_head(root))
            (root/'.git/HEAD').write_text('f'*9000,encoding='utf-8')
            self.assertIsNone(read_git_head(root))

    def test_lm_eval_git_cli_invalid_output_falls_back_and_normalizes_declared_revision(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'lm_eval').mkdir(); (root/'.git').mkdir()
            (root/'.git/HEAD').write_text('E'*40+'\n',encoding='utf-8')
            invalid=Mock(returncode=0,stdout='not-a-git-object\n')
            with patch.object(lm_eval_impl.subprocess,'run',return_value=invalid):
                _,revision,dirty=lm_eval_impl._framework_source({
                    'tool_root':str(root),'provenance_policy':'strict',
                    'framework_revision':'E'*40,'require_clean_framework':False,
                })
            self.assertEqual(revision,'e'*40)
            self.assertIsNone(dirty)

    def test_both_lm_eval_bindings_use_git_metadata_fallback(self):
        import importlib.util
        from model_evaluation.adapters.binding.lm_eval import impl as generic_impl
        from unittest.mock import patch
        module_path=PACKAGE_ROOT/'adapters'/'binding'/'lm_eval.bbh'/'impl.py'
        spec=importlib.util.spec_from_file_location('test_lm_eval_bbh_gitmeta_impl',module_path)
        self.assertIsNotNone(spec); self.assertIsNotNone(spec.loader)
        bbh_impl=importlib.util.module_from_spec(spec); spec.loader.exec_module(bbh_impl)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); (root/'.git/refs/heads').mkdir(parents=True)
            (root/'.git/HEAD').write_text('ref: refs/heads/main\n',encoding='utf-8')
            (root/'.git/refs/heads/main').write_text('a'*40+'\n',encoding='utf-8')
            evaluation={'parameters':{'framework_revision':'A'*40,'provenance_policy':'strict'}}
            with patch.object(generic_impl.subprocess,'run',side_effect=FileNotFoundError('git')):
                self.assertEqual(generic_impl._framework_revision(root,evaluation),'a'*40)
            with patch.object(bbh_impl.subprocess,'run',side_effect=FileNotFoundError('git')):
                self.assertEqual(bbh_impl._revision(root,evaluation),'a'*40)


    def test_openai_generation_probe_is_independent_from_likelihood_features(self):
        from unittest.mock import patch
        from model_evaluation.sdk import openai_service
        def fake_request(url, method='GET', payload=None, **kwargs):
            if url.endswith('/models'): return 200, {'data':[{'id':'m'}]}
            if url.endswith('/completions') and payload and 'logprobs' not in payload: return 200, {'choices':[{'text':' world'}]}
            raise AdapterError('SERVICE_NOT_READY','HTTP 400: logprobs unsupported',retryable=False)
        def fake_probe(url, method='GET', payload=None, **kwargs):
            if url.endswith('/completions'): return False, None, 'HTTP 400: logprobs unsupported'
            return False, None, 'not available'
        with patch.object(openai_service,'request_json',side_effect=fake_request), patch.object(openai_service,'optional_json_probe_detail',return_value=(False,None,'HTTP 400: logprobs unsupported',False)), patch.object(openai_service,'optional_json_probe',side_effect=fake_probe):
            desc=openai_service.probe_openai_service(base_url='http://127.0.0.1:1/v1',model='m',ownership='external',auth={'mode':'none'},bearer=None,timeout=1)
        caps=desc['capabilities']['values']
        self.assertTrue(caps['service.generation'])
        self.assertFalse(caps['service.completion_logprobs'])
        self.assertFalse(caps['service.echo'])


    def test_doctor_output_redacts_resolved_secrets(self):
        import contextlib, io, runpy
        ns=runpy.run_path(str(PACKAGE_ROOT/'cli.py'))
        secret='doctor-secret-value'
        store=SecretStore({'secret://doctor':secret}); store.resolve('secret://doctor')
        orch=SimpleNamespace(pm=SimpleNamespace(secrets=store))
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            ns['doctor_dump']({'stderr':f'failed token={secret}'},orch)
        rendered=out.getvalue()
        self.assertNotIn(secret,rendered)
        self.assertIn('<redacted>',rendered)

    def test_current_environment_resolve_uses_controller_python_context(self):
        app=Application(PACKAGE_ROOT, ROOT)
        client=app.registry.get('environment','current')
        expected=str(Path(sys.executable).resolve())
        out=client.invoke(
            'resolve',
            {'profile':'current','parameters':{}},
            context={'controller_python':'/caller-cannot-override'},
        )
        self.assertEqual(out['python'],expected)
        self.assertEqual(out['executable_root'],str(Path(expected).parent))

    def test_current_environment_snapshot_probes_the_recorded_python(self):
        from model_evaluation.adapters.environment.current import impl as current_env
        from unittest.mock import patch
        observed={}
        def fake_run(argv,**_kwargs):
            observed['argv']=argv
            return SimpleNamespace(
                returncode=0,
                stdout='{"python_implementation":"CPython","python_version":"9.8.7"}\n',
            )
        with patch.object(current_env.subprocess,'run',side_effect=fake_run):
            out=current_env.snapshot({}, {'controller_python':sys.executable,'timeout_seconds':2})
        self.assertEqual(observed['argv'][0],str(Path(sys.executable).resolve()))
        self.assertEqual(out['python_version'],'9.8.7')
        self.assertEqual(out['python_implementation'],'CPython')

    def test_process_start_interrupt_during_identity_probe_does_not_leave_child(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            spec={"schema_version":"1.0","argv":[sys.executable,"-c","import time; time.sleep(30)"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            started=[]; real_popen=subprocess.Popen
            def capture(*args,**kwargs):
                proc=real_popen(*args,**kwargs); started.append(proc); return proc
            with patch('model_evaluation.core.process.manager.subprocess.Popen',side_effect=capture), \
                 patch('model_evaluation.core.process.manager._proc_start_ticks',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    pm.start(spec)
            self.assertEqual(len(started),1)
            deadline=time.time()+3
            while started[0].poll() is None and time.time()<deadline: time.sleep(0.05)
            self.assertIsNotNone(started[0].poll())

    def test_process_start_pgid_probe_failure_does_not_leave_child(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas,ownership_root=Path(td).resolve()/'proc')
            spec={"schema_version":"1.0","argv":[sys.executable,"-c","import time; time.sleep(30)"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            started=[]; real_popen=subprocess.Popen
            def capture(*args,**kwargs):
                proc=real_popen(*args,**kwargs); started.append(proc); return proc
            calls={'n':0}
            def transient_pgid(pid):
                calls['n']+=1
                if calls['n']==1: raise OSError('pgid probe failed')
                return os.getpgid(pid)
            with patch('model_evaluation.core.process.manager.subprocess.Popen',side_effect=capture), \
                 patch('model_evaluation.core.process.manager._proc_pgid',side_effect=transient_pgid):
                with self.assertRaisesRegex(ProcessError,'process start was not committed'):
                    pm.start(spec)
            deadline=time.time()+3
            while started[0].poll() is None and time.time()<deadline: time.sleep(0.05)
            self.assertIsNotNone(started[0].poll())

    def test_process_start_persistent_pgid_probe_failure_is_explicitly_incomplete(self):
        from unittest.mock import patch
        pm=ProcessManager(self.schemas)
        spec={"schema_version":"1.0","argv":[sys.executable,"-c","import time; time.sleep(30)"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
        started=[]; real_popen=subprocess.Popen
        def capture(*args,**kwargs):
            proc=real_popen(*args,**kwargs); started.append(proc); return proc
        try:
            with patch('model_evaluation.core.process.manager.subprocess.Popen',side_effect=capture), \
                 patch('model_evaluation.core.process.manager._proc_pgid',side_effect=OSError('pgid unavailable')):
                with self.assertRaises(ProcessError) as ctx:
                    pm.start(spec)
            self.assertEqual(ctx.exception.details.get('cleanup_status'),'incomplete')
        finally:
            for proc in started:
                if proc.poll() is None: proc.kill()
                try: proc.communicate(timeout=2)
                except Exception: pass

    def test_process_stdio_partial_setup_failure_closes_prior_handle(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            pm=ProcessManager(self.schemas)
            opened=(Path(td)/'opened.log').open('ab')
            spec={"schema_version":"1.0","argv":[sys.executable,"-V"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            with patch.object(pm,'_stdio',side_effect=[(opened,opened),OSError('stderr open failed')]):
                with self.assertRaisesRegex(ProcessError,'failed to prepare/start process'):
                    pm.start(spec)
            self.assertTrue(opened.closed)

    def test_release_staging_filters_host_metadata_and_rejects_symlinks(self):
        import runpy
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'project'; root.mkdir()
            results=root/'results'; results.mkdir(); marker=results/'result.json'; marker.write_text('{}\n')
            source=root/'model_evaluation'/'core'; source.mkdir(parents=True); (source/'app.py').write_text('# runtime\n')
            apple_double=source/'._app.py'; apple_double.write_bytes(b'\x00AppleDouble')
            schemas=root/'model_evaluation'/'schemas'; schemas.mkdir(parents=True)
            schema=schemas/'adapter_manifest.schema.json'; schema.write_text('{}\n')
            schema_apple_double=schemas/'._adapter_manifest.schema.json'; schema_apple_double.write_bytes(b'\x00AppleDouble')
            nested_metadata=source/'._metadata'; nested_metadata.mkdir(); (nested_metadata/'payload.py').write_text('SECRET = True\n')
            finder_metadata=root/'.DS_Store'; finder_metadata.write_bytes(b'Finder metadata')
            stage=Path(td)/'stage'
            release=runpy.run_path(str(ROOT/'scripts'/'build_release.py'))
            calls=[]
            prepare=release['prepare_release_stage']
            prepare.__globals__['validate_tree']=lambda path: calls.append(('validate',path))
            prepare.__globals__['check_installable_bundle']=lambda path: calls.append(('wheel',path))
            prepare(root,stage)
            ignored_link=results/'ignored-link'; ignored_link.symlink_to(source/'app.py')
            release['reject_release_symlinks'](root,label='release source tree',ignore_excluded=True)
            self.assertTrue(marker.is_file())
            self.assertTrue(apple_double.is_file())
            self.assertTrue(finder_metadata.is_file())
            self.assertFalse((stage/'results').exists())
            self.assertFalse((stage/'.DS_Store').exists())
            self.assertFalse((stage/'model_evaluation'/'core'/'._app.py').exists())
            self.assertFalse((stage/'model_evaluation'/'core'/'._metadata').exists())
            self.assertTrue((stage/'model_evaluation'/'core'/'app.py').is_file())
            self.assertEqual(calls,[('validate',stage),('wheel',stage)])
            released=list(release['files_under'](stage))
            self.assertIn(stage/'model_evaluation'/'schemas'/'adapter_manifest.schema.json',released)
            self.assertNotIn(stage/'model_evaluation'/'core'/'._metadata'/'payload.py',released)

            unsafe=root/'linked-runtime.py'; unsafe.symlink_to(source/'app.py')
            with self.assertRaisesRegex(SystemExit,'release source tree contains symlink: linked-runtime.py'):
                release['reject_release_symlinks'](root,label='release source tree',ignore_excluded=True)
            with self.assertRaisesRegex(SystemExit,'release staging tree contains symlink: linked-runtime.py'):
                prepare(root,Path(td)/'unsafe-stage')

    def test_current_environment_wrap_uses_resolved_python(self):
        from model_evaluation.adapters.environment.current import impl as current_env
        env={'python':'/resolved/python3'}
        proc={'schema_version':'1.0','argv':['python','-m','lm_eval']}
        out=current_env.wrap({'process':proc,'environment':env},{})['process']
        self.assertEqual(out['argv'][0],'/resolved/python3')
        self.assertEqual(proc['argv'][0],'python')

    def test_lm_eval_normalize_rejects_wrong_single_task_identity(self):
        import json
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            Path(td,'result.json').write_text(json.dumps({'results':{'wrong_task':{'acc,none':0.9,'acc_stderr,none':0.01}}}),encoding='utf-8')
            with self.assertRaises(AdapterError):
                lm_eval_impl.normalize({'raw_result_root':td,'task':{'task_id':'expected_task','protocol_fingerprint':'pf'},'run_metadata':{'run_id':'r','model':'m','benchmark':'b'}},{})

    def test_lm_eval_metric_contract_maps_framework_metric_to_canonical(self):
        import json
        from model_evaluation.adapters.binding.lm_eval import impl as binding_impl
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        benchmark={'id':'mmlu','metrics':['accuracy']}
        evaluation={'parameters':{'metric_maps':{'mmlu':{'acc,none':'accuracy'}}}}
        contract=binding_impl._metric_contract(benchmark,evaluation)
        self.assertEqual(contract['metric_namespace'],'canonical')
        with tempfile.TemporaryDirectory() as td:
            Path(td,'result.json').write_text(json.dumps({'groups':{'mmlu':{'acc,none':0.75,'acc_stderr,none':0.05}}}),encoding='utf-8')
            task={'task_id':'mmlu','protocol_fingerprint':'pf','metrics':{'namespace':contract['metric_namespace'],'required':contract['required_metrics'],'mapping':contract['metric_map']}}
            out=lm_eval_impl.normalize({'raw_result_root':td,'task':task,'run_metadata':{'run_id':'r','model':'m','benchmark':'mmlu'}},{})
        self.assertEqual(out['metrics'],{'accuracy':{'value':0.75,'stderr':0.05}})
        self.assertEqual(out['metadata']['metric_namespace'],'canonical')

    def test_openai_generation_transient_failure_is_retryable_not_unsupported(self):
        from unittest.mock import patch
        from model_evaluation.sdk import openai_service
        with patch.object(openai_service,'request_json',side_effect=AdapterError('SERVICE_NOT_READY','HTTP 503',retryable=True)):
            with self.assertRaises(AdapterError) as cm:
                openai_service.probe_completion_capabilities(base='http://127.0.0.1:1/v1',model='m',bearer=None,timeout=1)
        self.assertTrue(cm.exception.retryable)

    def test_lm_eval_runner_rebinds_inner_python_to_runner_interpreter(self):
        from model_evaluation.adapters.evaluator.lm_eval import runner
        out=runner._bind_python_to_current_interpreter(['python','-m','lm_eval'])
        self.assertEqual(out[0],sys.executable)

    def test_operation_contract_rejects_service_and_task_identity_mismatch(self):
        service={'schema_version':'1.0','service_type':'llm','ownership':'external','model':{'id':'wrong'},'protocols':{'reference':{'url':'https://example.invalid/reference'}},'capabilities':{'schema_version':'1.0','values':{}},'auth':{'mode':'none'}}
        with self.assertRaises(Exception):
            validate_operation_output(self.schemas,'backend','probe_service',service,input_obj={'attach':{'model_id':'expected','ownership':'external','auth':{'mode':'none'}}})
        task={'schema_version':'1.0','framework':'other','benchmark_id':'wrong','task_id':'t','protocol_fingerprint':'0'*64}
        with self.assertRaises(Exception):
            validate_operation_output(self.schemas,'binding','build_task',task,input_obj={'benchmark':{'id':'b'},'evaluation':{'framework':{'adapter':'lm_eval'}}})

    def test_benchmark_binding_override_is_explicit_not_name_inferred(self):
        import copy
        app=Application(PACKAGE_ROOT, ROOT)
        evaluation=copy.deepcopy(app.specs.resolve('evaluation','lm_eval_current'))
        evaluation['id']='lm-eval-with-wrong-default-binding'
        evaluation['binding']={'adapter':'reference_eval'}
        app.specs.register('evaluation',evaluation)
        benchmark=copy.deepcopy(app.specs.resolve('benchmark','mmlu'))
        benchmark['id']='mmlu-explicit-binding'
        benchmark['bindings']={'lm_eval':'lm_eval'}
        app.specs.register('benchmark',benchmark)
        plan=app.planner.build({
            'schema_version':'1.0','model':'qwen_example','platform':'evaluation_current',
            'deployment':'openai_external_example','benchmark':benchmark['id'],'evaluation':evaluation['id'],
        })
        self.assertEqual(plan['resolved']['binding_adapter'],'lm_eval')

    def test_adapter_operation_inputs_are_formally_schema_validated(self):
        validate_operation_input(self.schemas,'device','probe',{'requested_devices':['gpu-A'],'parameters':{}})
        with self.assertRaises(Exception):
            validate_operation_input(self.schemas,'device','probe',{'requested_devices':'gpu-A','parameters':{}})
        with self.assertRaises(Exception):
            validate_operation_input(self.schemas,'backend','probe_service',{'attach':'not-an-object'})

    def test_evaluator_requirements_are_confined_to_service_and_evaluation_environment(self):
        valid={"schema_version":"1.0","requirements":[{"path":"service.protocol.custom_rpc","op":"equals","value":True}]}
        validate_operation_output(self.schemas,'evaluator','requirements',valid,input_obj={})
        leaked={"schema_version":"1.0","requirements":[{"path":"backend.vllm.logprobs","op":"equals","value":True}]}
        with self.assertRaises(Exception):
            validate_operation_output(self.schemas,'evaluator','requirements',leaked,input_obj={})

    def test_service_descriptor_supports_non_openai_protocol_without_core_changes(self):
        service={
            "schema_version":"1.0","service_type":"llm","ownership":"external","model":{"id":"m"},
            "protocols":{"custom_rpc":{"url":"https://example.invalid/rpc"}},
            "capabilities":{"schema_version":"1.0","values":{"service.custom_feature":True}},
            "auth":{"mode":"none"},
        }
        self.schemas.validate('service_descriptor',service)
        facts=facts_from_service(service)
        requirements={"schema_version":"1.0","requirements":[
            {"path":"service.protocol.custom_rpc","op":"equals","value":True},
            {"path":"service.custom_feature","op":"equals","value":True},
        ]}
        self.assertTrue(evaluate(requirements,facts).compatible)

    def test_dataset_identity_is_bound_to_planning_resolution(self):
        with self.assertRaises(CompatibilityError):
            Orchestrator._verify_dataset_identity({'dataset_id':'mmlu','revision':'r1'},{'dataset':{'revision':'r1'}},{'dataset_id':'other','revision':'r1'})
        with self.assertRaises(CompatibilityError):
            Orchestrator._verify_dataset_identity({'dataset_id':'mmlu','revision':'r1'},{'dataset':{'revision':'r1'}},{'dataset_id':'mmlu','revision':'r2'})

    def test_vllm_only_publishes_observed_optional_protocols(self):
        from unittest.mock import patch
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        with patch.object(vllm_impl,'request_json',return_value=(200,{'data':[{'id':'m'}]})),              patch.object(vllm_impl,'probe_completion_capabilities',return_value=(False,False,False,{})),              patch.object(vllm_impl,'optional_json_probe',return_value=(False,None,'404')):
            desc=vllm_impl.probe_service({'attach':{'base_url':'http://127.0.0.1:1/v1','model_id':'m','ownership':'external','auth':{'mode':'none'}}},{'timeout_seconds':1})
        self.assertEqual(set(desc['protocols']),{'openai_models'})

    def test_likelihood_transient_probe_is_retryable(self):
        from unittest.mock import patch
        from model_evaluation.sdk import openai_service
        with patch.object(openai_service,'request_json',return_value=(200,{'choices':[{'text':' world'}]})), \
             patch.object(openai_service,'optional_json_probe_detail',return_value=(False,None,'HTTP 503',True)):
            with self.assertRaises(AdapterError) as cm:
                openai_service.probe_completion_capabilities(base='http://127.0.0.1:1/v1',model='m',bearer=None,timeout=1)
        self.assertTrue(cm.exception.retryable)

    def test_openai_model_mismatch_is_non_retryable(self):
        from model_evaluation.sdk.openai_service import _listed_model
        with self.assertRaises(AdapterError) as cm:
            _listed_model({'data':[{'id':'other'}]},'expected')
        self.assertFalse(cm.exception.retryable)

    def test_readiness_non_retryable_error_fails_immediately(self):
        class Client:
            calls=0
            def invoke(self,*args,**kwargs):
                self.calls += 1
                raise AdapterExecutionError('SERVICE_NOT_READY','permanent',retryable=False)
        client=Client()
        dummy=object.__new__(Orchestrator)
        with self.assertRaises(AdapterExecutionError):
            dummy._probe_service_until_ready(client,{'model_id':'m'},None,1.0)
        self.assertEqual(client.calls,1)


    def test_reference_evaluator_and_binding_complete_protocol_cycle(self):
        import subprocess
        app=Application(PACKAGE_ROOT, ROOT)
        benchmark=app.specs.resolve('benchmark','mmlu')
        evaluation=app.specs.resolve('evaluation','reference_eval_current')
        dataset_client=app.registry.get('dataset','virtual')
        binding=app.registry.get('binding','reference_eval')
        evaluator=app.registry.get('evaluator','reference_eval')
        with tempfile.TemporaryDirectory() as td:
            artifact=dataset_client.invoke('prepare',{'benchmark':benchmark,'cache_root':td},context={'cache_root':td})
            task=binding.invoke('build_task',{'benchmark':benchmark,'dataset_artifact':artifact,'evaluation':evaluation})
            self.assertEqual(task['framework'],'reference_eval')
            req=evaluator.invoke('requirements',{'task':task,'evaluation':evaluation})
            self.assertEqual(req['requirements'][0]['path'],'evaluation_environment.python')
            output_root=Path(td).resolve()/'raw'
            service={'schema_version':'1.0','service_type':'llm','ownership':'external','model':{'id':'m'},'protocols':{'reference':{'url':'https://example.invalid/reference'}},'capabilities':{'schema_version':'1.0','values':{}},'auth':{'mode':'none'}}
            planned=evaluator.invoke('plan_evaluate',{
                'service':service,'task':task,'evaluation':evaluation,
                'output_root':str(output_root),'workspace':str(Path(td)),
                'log_path':str(Path(td)/'eval.log'),'network_policy':'offline',
            })
            argv=list(planned['process']['argv']); argv[0]=sys.executable
            proc=subprocess.run(argv,cwd=planned['process'].get('cwd'),check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            result=evaluator.invoke('normalize',{
                'raw_result_root':planned['raw_result_root'],'task':task,
                'run_metadata':{'run_id':'run-reference','model':'model-reference','benchmark':'mmlu'},
            })
            self.assertEqual(result['framework'],'reference_eval')
            self.assertEqual(result['metrics']['contract_ok']['value'],1)

    def test_lm_eval_snapshot_version_comes_from_manifest_identity(self):
        app=Application(PACKAGE_ROOT, ROOT)
        client=app.registry.get('evaluator','lm_eval')
        evaluation=app.specs.resolve('evaluation','lm_eval_current')
        evaluation['parameters']['tool_root']='/definitely/missing'
        snap=client.invoke('snapshot',{'evaluation':evaluation})
        self.assertEqual(snap['adapter_version'],client.identity.version)


    def test_execution_revalidates_full_requirements_not_only_identity(self):
        class FakeClient:
            def __init__(self, kind, name, handlers):
                self.identity=SimpleNamespace(kind=kind,name=name)
                self.handlers=handlers; self.last_warnings=[]
            def invoke(self, operation, input_obj, *, context=None, timeout=None):
                self.last_warnings=[]
                return self.handlers[operation](input_obj)
        expected_device={"schema_version":"1.0","vendor":"fake","device_type":"gpu","devices":[{"id":"0","name":"gpu0"}],"capabilities":{"schema_version":"1.0","values":{}}}
        expected_runtime={"schema_version":"1.0","family":"fake-runtime","version":"1.0","available":True,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["fake"],"runtime.feature_x":True}},"env_patch":{}}
        fresh_runtime={**expected_runtime,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["fake"],"runtime.feature_x":False}}}
        eval_env={"schema_version":"1.0","provider":"fake-env","identity":"eval","python":"/usr/bin/python","capabilities":{"schema_version":"1.0","values":{"environment.python":True}},"metadata":{}}
        backend_env={"schema_version":"1.0","provider":"fake-env","identity":"backend","python":"/usr/bin/python","capabilities":{"schema_version":"1.0","values":{"environment.python":True}},"metadata":{}}
        clients={
            ('device','fake-device'):FakeClient('device','fake-device',{'probe':lambda i:expected_device,'visibility':lambda i:{'env_patch':{}}}),
            ('runtime','fake-runtime'):FakeClient('runtime','fake-runtime',{'probe':lambda i:fresh_runtime,'resolve_environment':lambda i:{'env_patch':{}}}),
            ('environment','fake-env'):FakeClient('environment','fake-env',{'resolve':lambda i: eval_env if i['profile']=='eval' else backend_env}),
        }
        registry=SimpleNamespace(get=lambda kind,name:clients[(kind,name)])
        orch=Orchestrator(project_root=PACKAGE_ROOT,schemas=self.schemas,registry=registry,process_manager=None,resource_manager=None,results_root='/tmp',cache_root='/tmp')
        plan={'resolved':{'specs':{
            'deployment':{'management':{'mode':'managed'}},
            'platform':{'device':{'adapter':'fake-device','devices':['0']},'runtime':{'adapter':'fake-runtime'},'backend_environment':{'provider':'fake-env','profile':'backend'},'evaluation_environment':{'provider':'fake-env','profile':'eval'}},
        },'platform':{'device':expected_device,'runtime':expected_runtime,'backend_environment':backend_env,'evaluation_environment':eval_env,'device_env_patch':{},'runtime_env_patch':{}},
        'backend_requirements':{'schema_version':'1.0','requirements':[{'path':'runtime.feature_x','op':'equals','value':True}]},
        'binding_requirements':{'schema_version':'1.0','requirements':[]},
        'deployment_compatibility_requirements':{'schema_version':'1.0','requirements':[{'path':'runtime.family','op':'equals','value':'fake-runtime'}]}}}
        with tempfile.TemporaryDirectory() as td:
            run_dir=Path(td)
            with self.assertRaisesRegex(CompatibilityError,'backend requirements changed'):
                orch._revalidate_platform(plan,run_dir)
            diagnostic=json_loads_strict((run_dir/'.run'/'diagnostics'/'execution_preflight_compatibility.json').read_text())
            self.assertFalse(diagnostic['compatible'])

    def test_venv_environment_adapter_wraps_process_without_global_python(self):
        from model_evaluation.adapters.environment.venv import impl as venv_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'venv'; bindir=root/'bin'; bindir.mkdir(parents=True)
            py=bindir/'python'; py.symlink_to(Path(sys.executable).resolve())
            desc=venv_impl.resolve({'profile':str(root)},{})
            wrapped=venv_impl.wrap({'process':{'schema_version':'1.0','argv':['python','-c','print(1)'],'env_patch':{}},'environment':desc},{})['process']
            self.assertEqual(wrapped['argv'][0],str(py))
            self.assertEqual(wrapped['env_patch']['set']['VIRTUAL_ENV'],str(root.resolve()))
            self.assertEqual(wrapped['env_patch']['prepend_path']['PATH'][0],str(bindir.resolve()))

    def test_lm_eval_plan_preflight_is_a_formal_process_contract(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); pkg=root/'lm_eval'; pkg.mkdir(); (pkg/'__init__.py').write_text('VALUE = 1\n')
            evaluation={'schema_version':'1.1','id':'e','framework':{'adapter':'lm_eval'},'binding':{'adapter':'lm_eval'},'parameters':{'tool_root':str(root),'require_clean_framework':False}}
            task={'schema_version':'1.0','framework':'lm_eval','benchmark_id':'b','task_id':'b','artifacts':[],'protocol_fingerprint':'0'*64}
            input_obj={'evaluation':evaluation,'task':task}
            validate_operation_input(self.schemas,'evaluator','plan_preflight',input_obj)
            out=lm_eval_impl.plan_preflight(input_obj,{})
            validate_operation_output(self.schemas,'evaluator','plan_preflight',out,input_obj=input_obj)
            self.assertEqual(out['result_format'],'preflight_result')
            payload=json.loads(out['process']['argv'][-1])
            self.assertEqual(payload['task_id'],'b')
            self.assertEqual(out['process']['argv'][0],'python')
            self.assertIn(str(root.resolve()),out['process']['argv'][-1])

    def test_lm_eval_preflight_and_evaluation_share_explicit_cache_environment(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'harness'; pkg=root/'lm_eval'; pkg.mkdir(parents=True); (pkg/'__init__.py').write_text('VALUE = 1\n')
            cache=Path(td)/'cache'
            evaluation={'schema_version':'1.1','id':'e','framework':{'adapter':'lm_eval'},'binding':{'adapter':'lm_eval'},'parameters':{'tool_root':str(root),'require_clean_framework':False}}
            task={'schema_version':'1.0','framework':'lm_eval','benchmark_id':'b','task_id':'b','artifacts':[],'protocol_fingerprint':'0'*64}
            pre=lm_eval_impl.plan_preflight({'evaluation':evaluation,'task':task,'cache_root':str(cache)},{'offline':True})['process']
            service={'schema_version':'1.0','service_type':'llm','ownership':'external','model':{'id':'m'},'protocols':{'openai_completion':{'url':'http://127.0.0.1:1/v1/completions'}},'capabilities':{'schema_version':'1.0','values':{}},'auth':{'mode':'none'}}
            planned=lm_eval_impl.plan_evaluate({'service':service,'task':task,'evaluation':evaluation,'cache_root':str(cache),'output_root':str(Path(td)/'out'),'network_policy':'offline'},{'offline':True})['process']
            expected={
                'HF_HOME':str(cache/'huggingface'),
                'HF_DATASETS_CACHE':str(cache/'huggingface'/'datasets'),
                'HF_HUB_CACHE':str(cache/'huggingface'/'hub'),
                'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            }
            for name,value in expected.items():
                self.assertEqual(pre['env_patch']['set'][name],value)
                self.assertEqual(planned['env_patch']['set'][name],value)

    def test_lm_eval_task_preflight_reports_missing_offline_data_structurally(self):
        helper=PACKAGE_ROOT/'adapters'/'evaluator'/'lm_eval'/'preflight.py'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); pkg=root/'lm_eval'; tasks=pkg/'tasks'; tasks.mkdir(parents=True)
            (pkg/'__init__.py').write_text('VALUE = 1\n')
            (tasks/'__init__.py').write_text(
                'class TaskManager:\n'
                '    def __init__(self, include_path=None): pass\n'
                '    def load(self, tasks):\n'
                '        raise ConnectionError("Could not reach dataset because OfflineModeIsEnabled")\n'
            )
            payload=json.dumps({'framework_root':str(root),'task_id':'missing'})
            proc=subprocess.run([sys.executable,str(helper),'--payload',payload],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
            self.assertEqual(proc.returncode,2,proc.stderr)
            result=json.loads(proc.stdout)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['error']['code'],'EVALUATION_DATA_UNAVAILABLE')

    def test_backend_snapshot_does_not_probe_controller_environment_for_workload_version(self):
        from model_evaluation.adapters.backend.vllm import impl as vllm_impl
        snap=vllm_impl.snapshot({'deployment':{'parameters':{'executable':'definitely-not-on-controller-path'}}},{})
        self.assertEqual(snap['configured_executable'],'definitely-not-on-controller-path')
        self.assertEqual(snap['version_source'],'selected_environment_probe')
        self.assertNotIn('version',snap)

    def test_conda_environment_supports_named_and_absolute_prefix_selection(self):
        from model_evaluation.adapters.environment.conda import impl as conda_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); fake_conda=root/'conda'; fake_conda.write_text('#!/bin/sh\nexit 0\n'); fake_conda.chmod(0o755)
            prefix=root/'envs'/'backend'; (prefix/'bin').mkdir(parents=True)
            named_env={'schema_version':'1.0','provider':'conda','identity':'backend-name','python':'/tmp/python','executable_root':'/tmp'}
            named=conda_impl.wrap({'process':{'schema_version':'1.0','argv':['vllm','--version'],'env_patch':{}},'environment':named_env,'parameters':{'executable':str(fake_conda)}},{})['process']
            self.assertEqual(named['argv'][:5],[str(fake_conda.resolve()),'run','--no-capture-output','-n','backend-name'])
            prefix_env={'schema_version':'1.0','provider':'conda','identity':str(prefix),'python':str(prefix/'bin'/'python'),'executable_root':str(prefix/'bin')}
            prefixed=conda_impl.wrap({'process':{'schema_version':'1.0','argv':['vllm','--version'],'env_patch':{}},'environment':prefix_env,'parameters':{'executable':str(fake_conda)}},{})['process']
            self.assertEqual(prefixed['argv'][:5],[str(fake_conda.resolve()),'run','--no-capture-output','-p',str(prefix.resolve())])

    def test_conda_resolve_uses_prefix_mode_for_absolute_profile(self):
        from model_evaluation.adapters.environment.conda import impl as conda_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); prefix=root/'envs'/'backend'; (prefix/'bin').mkdir(parents=True)
            (prefix/'bin'/'python').symlink_to(Path(sys.executable).resolve())
            fake_conda=root/'conda'
            fake_conda.write_text('#!/bin/sh\nexit 73\n')
            fake_conda.chmod(0o755)
            desc=conda_impl.resolve({'profile':str(prefix),'parameters':{'executable':str(fake_conda)}},{'timeout_seconds':2})
            self.assertEqual(desc['identity'],str(prefix.resolve()))
            self.assertEqual(desc['python'],str(prefix.resolve()/'bin'/'python'))
            self.assertEqual(desc['metadata']['selection_mode'],'prefix')
            self.assertEqual(desc['metadata']['probe_mode'],'direct_prefix')

    def test_isolated_environment_rejects_absolute_executable_from_another_environment(self):
        from model_evaluation.adapters.environment.venv import impl as venv_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); env_root=root/'env'; bindir=env_root/'bin'; bindir.mkdir(parents=True)
            (bindir/'python').symlink_to(Path(sys.executable).resolve())
            desc=venv_impl.resolve({'profile':str(env_root)},{})
            with self.assertRaisesRegex(Exception,'outside selected venv executable_root'):
                venv_impl.wrap({'process':{'schema_version':'1.0','argv':['/usr/bin/true'],'env_patch':{}},'environment':desc},{})

    def test_evaluator_preflight_contract_allows_doctor_without_task_materialization(self):
        from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval_impl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); pkg=root/'lm_eval'; pkg.mkdir(); (pkg/'__init__.py').write_text('VALUE = 1\n')
            evaluation={'schema_version':'1.1','id':'e','framework':{'adapter':'lm_eval'},'binding':{'adapter':'lm_eval'},'parameters':{'tool_root':str(root),'require_clean_framework':False}}
            input_obj={'evaluation':evaluation}
            validate_operation_input(self.schemas,'evaluator','plan_preflight',input_obj)
            out=lm_eval_impl.plan_preflight(input_obj,{})
            validate_operation_output(self.schemas,'evaluator','plan_preflight',out,input_obj=input_obj)
            self.assertEqual(out['process']['argv'][0],'python')
            self.assertEqual(out['process']['metadata']['scope'],'dependency')

    def test_stale_recovery_never_signals_legacy_record_without_boot_id(self):
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'proc'; root.mkdir()
            record=root/'process-123-x.json'
            record.write_text(json.dumps({'pid':123,'pgid':123,'start_ticks':9}))
            pm=ProcessManager(self.schemas,ownership_root=root)
            with patch('model_evaluation.core.process.manager._linux_boot_id',return_value='current-boot'), \
                 patch('model_evaluation.core.process.manager.os.killpg') as killpg:
                rows=pm.recover_stale_managed()
            self.assertEqual(rows[0]['status'],'invalid')
            self.assertIn('missing boot_id',rows[0]['error'])
            self.assertTrue(record.exists())
            killpg.assert_not_called()

    def test_stale_recovery_discards_record_from_previous_boot_without_signalling(self):
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'proc'; root.mkdir()
            record=root/'process-123-x.json'
            record.write_text(json.dumps({'pid':123,'pgid':123,'start_ticks':9,'boot_id':'old-boot'}))
            pm=ProcessManager(self.schemas,ownership_root=root)
            with patch('model_evaluation.core.process.manager._linux_boot_id',return_value='new-boot'), \
                 patch('model_evaluation.core.process.manager._proc_start_ticks',return_value=9), \
                 patch('model_evaluation.core.process.manager.os.killpg') as killpg:
                rows=pm.recover_stale_managed()
            self.assertEqual(rows[0]['status'],'expired_boot')
            self.assertFalse(record.exists())
            killpg.assert_not_called()

    def test_process_ownership_record_contains_boot_id(self):
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()/'proc'
            pm=ProcessManager(self.schemas,ownership_root=root)
            spec={"schema_version":"1.0","argv":[sys.executable,"-c","import time; time.sleep(30)"],"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"}}
            with patch('model_evaluation.core.process.manager._linux_boot_id',return_value='boot-test-id'):
                h=pm.start(spec)
            try:
                records=list(root.glob('process-*.json'))
                self.assertEqual(len(records),1)
                obj=json.loads(records[0].read_text())
                self.assertEqual(obj['boot_id'],'boot-test-id')
            finally:
                pm.stop(h,grace_seconds=0.05,kill_seconds=1.0)

    @staticmethod
    def _write_fake_llama_server(path: Path) -> None:
        script = """#!/usr/bin/env python3
import argparse, json, signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

if '--version' in sys.argv:
    print('fake-llama-server 1.0')
    raise SystemExit(0)

ap=argparse.ArgumentParser(add_help=False)
ap.add_argument('--model')
ap.add_argument('--alias', required=True)
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--ctx-size')
ap.add_argument('--wrong-model', action='store_true')
args,_=ap.parse_known_args()
model_id='wrong-model' if args.wrong_model else args.alias

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, obj):
        raw=json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path == '/v1/models':
            self._send(200, {'data':[{'id':model_id,'context_length':4096}]})
        else:
            self._send(404, {'error':'not found'})
    def do_POST(self):
        n=int(self.headers.get('Content-Length') or 0)
        payload=json.loads(self.rfile.read(n) or b'{}')
        if self.path == '/v1/completions':
            prompt=str(payload.get('prompt') or '')
            choice={'text': prompt+'x' if payload.get('echo') else 'x'}
            if payload.get('logprobs') is not None:
                choice['logprobs']={'token_logprobs':[0.0]}
            self._send(200, {'choices':[choice]})
        elif self.path == '/v1/chat/completions':
            self._send(200, {'choices':[{'message':{'role':'assistant','content':'x'}}]})
        else:
            self._send(404, {'error':'not found'})
    def log_message(self, *_args):
        pass

server=HTTPServer((args.host,args.port), Handler)
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
server.serve_forever()
"""
        path.write_text(script,encoding='utf-8')
        path.chmod(0o755)

    def _managed_e2e_plan(
        self,
        app: Application,
        root: Path,
        *,
        wrong_model: bool,
        port: int | None = None,
    ) -> dict:
        import socket
        if port is None:
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
                sock.bind(('127.0.0.1',0)); port=int(sock.getsockname()[1])
        model_file=root/'model.gguf'; model_file.write_bytes(b'fake-model')
        executable=root/'fake-llama-server'; self._write_fake_llama_server(executable)
        app.specs.register('model',{
            'schema_version':'1.0','id':'e2e_model','source':{'type':'local','ref':str(model_file)},
            'experiment_id':'e2e-catalog-model','label':'E2E Catalog Model',
            'provenance':{'policy':'migration'},
        })
        app.specs.register('platform',{
            'schema_version':'1.1','id':'e2e_platform',
            'device':{'adapter':'cpu'},'runtime':{'adapter':'cpu'},
            'backend_environment':{'provider':'current','profile':'current'},
            'evaluation_environment':{'provider':'current','profile':'current'},
        })
        params={
            'port':port,'context_length':4096,'num_concurrent':1,
            'ready_timeout_seconds':3,'shutdown_timeout_seconds':1,
            'executable':str(executable),
        }
        if wrong_model: params['extra_args']=['--wrong-model']
        app.specs.register('deployment',{
            'schema_version':'1.1','id':'e2e_deployment','backend':{'adapter':'llama_cpp'},
            'management':{'mode':'managed'},'model_location':{'local_path':str(model_file)},
            'parameters':params,'compatibility':{'runtime_families':['cpu']},
        })
        app.specs.register('benchmark',{
            'schema_version':'1.0','id':'e2e_benchmark','dataset':{'provider':'virtual','revision':'e2e'},
            'protocol':{'task':'e2e','fewshot':0,'inference':['multiple_choice']},'metrics':['contract_ok'],
            'bindings':{'reference_eval':'reference_eval'},
        })
        app.specs.register('evaluation',{
            'schema_version':'1.1','id':'e2e_evaluation','framework':{'adapter':'reference_eval'},
            'binding':{'adapter':'reference_eval'},
        })
        run={
            'schema_version':'1.0','model':'e2e_model','platform':'e2e_platform',
            'deployment':'e2e_deployment','benchmark':'e2e_benchmark','evaluation':'e2e_evaluation',
        }
        return app.planner.build(run)

    def test_orchestrator_managed_e2e_success(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            with patch.dict(os.environ,{'MODEL_EVAL_RUNTIME_ROOT':str(root/'runtime')},clear=False):
                app=Application(PACKAGE_ROOT, ROOT)
                plan=self._managed_e2e_plan(app,root,wrong_model=False)
                run_dir=app.orchestrator(results_root=root/'results',cache_root=root/'cache').execute(plan)
            result=json_loads_strict((run_dir/'result.json').read_text())
            metrics=json_loads_strict((run_dir/'metrics.json').read_text())
            terminal=json_loads_strict((run_dir/'terminal.json').read_text())
            run_config=json_loads_strict((run_dir/'config'/'run_config.json').read_text())
            runtime_versions=json_loads_strict((run_dir/'config'/'runtime_versions.json').read_text())
            self.assertEqual(result['model'],'e2e-catalog-model')
            self.assertEqual(metrics['model'],'e2e-catalog-model')
            self.assertEqual(result['metrics']['contract_ok']['value'],1)
            self.assertEqual(metrics['summary']['contract_ok']['value'],1)
            self.assertEqual(terminal['cleanup']['status'],'clean')
            self.assertEqual(terminal['cleanup']['backend']['graceful']['result'],'success')
            self.assertEqual(terminal['outcome'],'success')
            self.assertTrue(run_config['adapters'])
            self.assertEqual(runtime_versions['backend']['adapter'],'llama_cpp')
            self.assertEqual(runtime_versions['backend']['version'],'fake-llama-server 1.0')
            self.assertEqual(runtime_versions['evaluator']['adapter'],'reference_eval')
            self.assertEqual(runtime_versions['runtime']['family'],'cpu')
            self.assertTrue(runtime_versions['environments']['backend']['python'])
            self.assertFalse((run_dir/'.run').exists())
            self.assertFalse((run_dir/'evidence').exists())
            self.assertFalse((run_dir/'SHA256SUMS').exists())
            self.assertTrue((run_dir/'raw'/'framework_result.json').is_file())
            self.assertFalse(list((root/'runtime'/'processes').glob('process-*.json')))

    def test_initial_persistence_failure_still_publishes_terminal_product(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            with patch.dict(os.environ,{'MODEL_EVAL_RUNTIME_ROOT':str(root/'runtime')},clear=False):
                app=Application(PACKAGE_ROOT, ROOT)
                plan=self._managed_e2e_plan(app,root,wrong_model=False,port=18091)
                plan['resources']=[]
                orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
                with patch.object(orch,'_persist_initial',side_effect=OSError('initial persistence failed')):
                    with self.assertRaisesRegex(OSError,'initial persistence failed'):
                        orch.execute(plan)

            runs=[path for path in (root/'results').iterdir() if path.is_dir()]
            self.assertEqual(len(runs),1)
            run_dir=runs[0]
            failure=json_loads_strict((run_dir/'failure.json').read_text())
            terminal=json_loads_strict((run_dir/'terminal.json').read_text())
            self.assertEqual(failure['stage'],'INITIALIZING')
            self.assertEqual(failure['primary_error']['message'],'initial persistence failed')
            self.assertEqual(terminal['outcome'],'failed')
            self.assertEqual(terminal['error'],failure['primary_error'])
            self.assertEqual(terminal['cleanup']['status'],'clean')

    def test_orchestrator_managed_e2e_readiness_failure_preserves_error_and_cleans_backend(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            with patch.dict(os.environ,{'MODEL_EVAL_RUNTIME_ROOT':str(root/'runtime')},clear=False):
                app=Application(PACKAGE_ROOT, ROOT)
                plan=self._managed_e2e_plan(app,root,wrong_model=True)
                orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
                with self.assertRaises(AdapterExecutionError) as ctx:
                    orch.execute(plan)
            run_dir=Path(ctx.exception.details['run_dir'])
            failure=json_loads_strict((run_dir/'failure.json').read_text())
            terminal=json_loads_strict((run_dir/'terminal.json').read_text())
            self.assertEqual(failure['primary_error']['code'],'SERVICE_NOT_READY')
            self.assertEqual(failure['stage'],'SERVICE_STARTING')
            self.assertEqual(terminal['cleanup']['status'],'clean')
            self.assertEqual(terminal['cleanup']['backend']['graceful']['result'],'success')
            self.assertEqual(terminal['outcome'],'failed')
            self.assertTrue((run_dir/'.run').is_dir())
            self.assertFalse((run_dir/'SHA256SUMS').exists())
            self.assertFalse(list((root/'runtime'/'processes').glob('process-*.json')))

    def test_evaluator_preflight_timeout_publishes_failure_and_matrix_continues(self):
        """A bounded evaluator probe failure is an ordinary per-run failure.

        This covers the complete ProcessManager -> Orchestrator -> MatrixExecutor
        path.  In particular, it guards against a cleanup/finalization regression
        silently leaving a run in ``EVALUATOR_PREFLIGHT`` or preventing a serial
        ``continue_on_error`` matrix from starting its next child plan.
        """
        from unittest.mock import patch

        class EvaluatorWithPreflight:
            def __init__(self, delegate):
                self.delegate=delegate
                manifest=copy.deepcopy(delegate.identity.manifest)
                manifest['operations']=[*manifest.get('operations',[]),'plan_preflight']
                self.identity=SimpleNamespace(
                    kind=delegate.identity.kind,
                    name=delegate.identity.name,
                    version=delegate.identity.version,
                    path=delegate.identity.path,
                    manifest=manifest,
                )
                self.last_warnings=[]
                self.preflight_calls=0

            def invoke(self, operation, input_obj, *, context=None, timeout=None):
                if operation == 'plan_preflight':
                    self.preflight_calls += 1
                    if self.preflight_calls == 1:
                        command='import time; time.sleep(2)'
                        process_timeout=0.05
                    else:
                        command='print(\'{"schema_version":"1.0","status":"passed","facts":{"probe":"ok"}}\')'
                        process_timeout=2
                    self.last_warnings=[]
                    return {
                        'process':{
                            'schema_version':'1.0','argv':['python','-c',command],'env_patch':{},
                            'stdin':{'mode':'null'},'stdout':{'mode':'capture'},'stderr':{'mode':'capture'},
                            'timeout_seconds':process_timeout,
                        },
                        'result_format':'preflight_result',
                    }
                value=self.delegate.invoke(operation,input_obj,context=context,timeout=timeout)
                self.last_warnings=list(self.delegate.last_warnings)
                return value

        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            with patch.dict(os.environ,{'MODEL_EVAL_RUNTIME_ROOT':str(root/'runtime')},clear=False):
                app=Application(PACKAGE_ROOT, ROOT)
                first=self._managed_e2e_plan(app,root,wrong_model=False)
                second_model=copy.deepcopy(app.specs.resolve('model','e2e_model'))
                second_model.update({'id':'e2e_model_2','experiment_id':'e2e-catalog-model-2','label':'E2E Catalog Model 2'})
                app.specs.register('model',second_model)
                matrix=app.matrix_planner.build({
                    'schema_version':'1.0','id':'evaluator_preflight_timeout_continue',
                    'models':['e2e_model','e2e_model_2'],
                    'platforms':[first['run_spec']['platform']],
                    'deployments':[first['run_spec']['deployment']],
                    'benchmarks':[first['run_spec']['benchmark']],
                    'evaluations':[first['run_spec']['evaluation']],
                    'execution':{'mode':'serial','continue_on_error':True,'max_runs':2},
                })
                real_get=app.registry.get
                evaluator=EvaluatorWithPreflight(real_get('evaluator','reference_eval'))

                def get_with_preflight(kind,name):
                    if (kind,name) == ('evaluator','reference_eval'):
                        return evaluator
                    return real_get(kind,name)

                with patch.object(app.registry,'get',side_effect=get_with_preflight):
                    batch_dir,summary=app.matrix_executor(
                        results_root=root/'results',cache_root=root/'cache'
                    ).execute(matrix)

            self.assertEqual(evaluator.preflight_calls,2)
            self.assertEqual(summary['planned'],2)
            self.assertEqual(summary['failed'],1)
            self.assertEqual(summary['success'],1)
            self.assertFalse(summary['hard_stop'])
            self.assertTrue(summary['continue_on_error'])
            rows=json_loads_strict((batch_dir/'batch_status.json').read_text())['runs']
            failed=next(row for row in rows if row['status']=='failed')
            succeeded=next(row for row in rows if row['status']=='success')
            failed_run=Path(failed['run_dir'])
            failure=json_loads_strict((failed_run/'failure.json').read_text())
            terminal=json_loads_strict((failed_run/'terminal.json').read_text())
            self.assertEqual(failure['stage'],'EVALUATOR_PREFLIGHT')
            self.assertEqual(failure['primary_error']['code'],'PROCESS_ERROR')
            self.assertIn('process timed out',failure['primary_error']['message'])
            self.assertEqual(failure['cleanup']['status'],'clean')
            self.assertEqual(failure['cleanup']['backend']['status'],'not_started')
            self.assertEqual(terminal['outcome'],'failed')
            self.assertEqual(terminal['cleanup']['status'],'clean')
            self.assertTrue(Path(succeeded['result_path']).is_file())
            self.assertFalse(list((root/'runtime'/'processes').glob('process-*.json')))


    def test_final_status_diagnostic_failure_is_logged_without_hiding_success(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            with patch.dict(os.environ,{'MODEL_EVAL_RUNTIME_ROOT':str(root/'runtime')},clear=False):
                app=Application(PACKAGE_ROOT, ROOT); plan=self._managed_e2e_plan(app,root,wrong_model=False)
                orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
                real_status=orch._status
                def flaky(run_dir,state,**extra):
                    if state == 'SUCCEEDED': raise OSError('final status unavailable')
                    return real_status(run_dir,state,**extra)
                with patch.object(orch,'_status',side_effect=flaky):
                    run_dir=orch.execute(plan)
            self.assertIn('final status unavailable',(run_dir/'logs'/'core_error.log').read_text())
            self.assertEqual(json_loads_strict((run_dir/'terminal.json').read_text())['outcome'],'success')
            self.assertFalse((run_dir/'SHA256SUMS').exists())

    def test_final_status_diagnostic_failure_preserves_primary_failure(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            with patch.dict(os.environ,{'MODEL_EVAL_RUNTIME_ROOT':str(root/'runtime')},clear=False):
                app=Application(PACKAGE_ROOT, ROOT); plan=self._managed_e2e_plan(app,root,wrong_model=True)
                orch=app.orchestrator(results_root=root/'results',cache_root=root/'cache')
                real_status=orch._status
                def flaky(run_dir,state,**extra):
                    if state == 'FINALIZED': raise OSError('final failure status unavailable')
                    return real_status(run_dir,state,**extra)
                with patch.object(orch,'_status',side_effect=flaky):
                    with self.assertRaises(AdapterExecutionError) as ctx:
                        orch.execute(plan)
            run_dir=Path(ctx.exception.details['run_dir'])
            self.assertTrue((run_dir/'logs'/'core_error.log').is_file())
            failure=json_loads_strict((run_dir/'failure.json').read_text())
            self.assertEqual(failure['primary_error']['code'],'SERVICE_NOT_READY')


if __name__=='__main__': unittest.main()
