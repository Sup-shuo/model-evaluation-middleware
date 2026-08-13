from __future__ import annotations
import argparse, copy, json, os, sys
from pathlib import Path
from model_evaluation import package_root
from model_evaluation.core.app import Application
from model_evaluation.core.files import atomic_json
from model_evaluation.core.errors import ModelEvalError
from model_evaluation.core.config.deployment import resolve_deployment_profile
from model_evaluation.core.config.evaluation import resolve_evaluation_profile
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.core.security import redact_diagnostic

EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS = 30

def dump(obj): print(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True))

def doctor_dump(obj, orch):
    dump(redact_diagnostic(obj, orch.pm.secrets.redaction_values()))

def add_user_config_args(p):
    p.add_argument('--system-config',default=None,help='机器配置路径或 config/systems/ 下的 ID；默认 MODEL_EVAL_SYSTEM_CONFIG 或 config/system.yaml')
    p.add_argument('--evaluation-config',default=None,help='评测配置路径或 config/evaluations/ 下的 ID；默认 MODEL_EVAL_EVALUATION_CONFIG 或 config/evaluation.yaml')

def _project_root() -> Path:
    configured = os.environ.get("MODEL_EVAL_PROJECT_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _main():
    ap=argparse.ArgumentParser(prog='eval-manager',description='模型评测适配中间层 v4.1')
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('schema-check'); sub.add_parser('adapters')
    p=sub.add_parser('validate',help='不带参数时验证 config/system.yaml + config/evaluation.yaml'); p.add_argument('run',nargs='?'); add_user_config_args(p)
    p=sub.add_parser('doctor',help='检查当前机器、所选环境与评测框架是否具备本地运行条件（不启动模型服务）'); add_user_config_args(p)
    p=sub.add_parser('plan',help='不带 RunSpec 时从两份用户配置生成批量计划'); p.add_argument('run',nargs='?'); p.add_argument('-o','--output'); add_user_config_args(p)
    p=sub.add_parser('run',help='不带 RunSpec 时直接运行 config/evaluation.yaml'); p.add_argument('run',nargs='?'); p.add_argument('--results-root'); p.add_argument('--cache-root'); add_user_config_args(p)
    p=sub.add_parser('run-plan'); p.add_argument('plan'); p.add_argument('--results-root'); p.add_argument('--cache-root'); p.add_argument('--continue-on-error',action=argparse.BooleanOptionalAction,default=None); p.add_argument('--resume-dir')
    p=sub.add_parser('matrix-validate'); p.add_argument('matrix')
    p=sub.add_parser('matrix-expand'); p.add_argument('matrix'); p.add_argument('-o','--output')
    p=sub.add_parser('matrix-plan'); p.add_argument('matrix'); p.add_argument('-o','--output')
    p=sub.add_parser('matrix-run'); p.add_argument('matrix'); p.add_argument('--results-root'); p.add_argument('--cache-root'); p.add_argument('--continue-on-error',action=argparse.BooleanOptionalAction,default=None)
    p=sub.add_parser('run-matrix-plan'); p.add_argument('plan'); p.add_argument('--results-root'); p.add_argument('--cache-root'); p.add_argument('--continue-on-error',action=argparse.BooleanOptionalAction,default=None); p.add_argument('--resume-dir')
    args=ap.parse_args(); app=Application(package_root(), project_root=_project_root())
    if args.cmd=='schema-check': dump({'ok':True,'schemas':app.schemas.validate_all_schemas()}); return
    if args.cmd=='adapters':
        app.registry.discover(); dump([x.manifest for x in app.registry.identities()]); return
    if args.cmd=='validate':
        if args.run:
            run=app.specs.resolve_run(args.run); bundle=app.specs.resolve_bundle(run); _,dep_resolution=resolve_deployment_profile(bundle['deployment'],bundle['model'],bundle['platform'],(run.get('overrides') or {}).get('deployment')); _,eval_resolution=resolve_evaluation_profile(bundle['evaluation'],bundle['platform']); dump({'ok':True,'mode':'internal-run-spec','run':run,'resolved_ids':{k:(v.get('id') if isinstance(v,dict) else None) for k,v in bundle.items()},'deployment_resolution':dep_resolution,'evaluation_resolution':eval_resolution}); return
        bundle=app.load_user_config(args.system_config,args.evaluation_config)
        dump({'ok':True,'mode':'user-config','system':bundle.system['system']['name'],'profiles':bundle.generated.get('selected_profiles',{}),'models':list(bundle.generated['model_ids'].values()),'benchmarks':bundle.evaluation['benchmarks'],'cache_root':bundle.cache_root,'results_root':bundle.results_root,'generated':bundle.generated}); return
    if args.cmd=='doctor':
        plan,bundle=app.user_matrix_plan(args.system_config,args.evaluation_config)
        orch=app.orchestrator(results_root=bundle.results_root,cache_root=bundle.cache_root)

        def text(value):
            if value is None: return ''
            return value.decode('utf-8','replace') if isinstance(value,(bytes,bytearray)) else str(value)

        def wrapped_process(process, *, platform_spec, resolved_platform, role, base_patches=()):
            return orch.prepare_process_for_environment(
                process,
                platform_spec=platform_spec,
                resolved_platform=resolved_platform,
                role=role,
                base_patches=tuple(base_patches),
                context={'doctor':True,'offline':True,'preflight':True},
                timeout=5,
            )


        rows=[]
        for child in plan['plans']:
            specs=child['resolved']['specs']; platform=specs['platform']; resolved_platform=child['resolved'].get('platform') or {}
            model=specs['model']; deployment=specs['deployment']; evaluator=specs['evaluation']; mode=deployment['management']['mode']
            meta=model.get('metadata') or {}
            row={
                'model_id':model.get('experiment_id') or meta.get('experiment_id') or model['id'],
                'model_label':model.get('label') or meta.get('label') or model.get('experiment_id') or meta.get('experiment_id') or model['id'],
                'model_ref':(model.get('source') or {}).get('ref'),
                'benchmark':specs['benchmark']['id'],
                'compatibility':child['compatibility']['status'],
                'reasons':child['compatibility'].get('reasons') or [],
                'backend_environment':(resolved_platform.get('backend_environment') or {}).get('identity'),
                'evaluation_environment':(resolved_platform.get('evaluation_environment') or {}).get('identity'),
                'runtime':(resolved_platform.get('runtime') or {}).get('family'),
                'devices':[d.get('id') for d in ((resolved_platform.get('device') or {}).get('devices') or [])],
                'checks':{},
                'deferred':['backend service readiness','service capability compatibility'],
                'warnings':copy.deepcopy(child.get('warnings') or []),
            }

            # Doctor keeps its evaluator probe dependency-only. The run path adds
            # the bound FrameworkTaskArtifact and verifies task/data readiness
            # before model startup, while doctor remains side-effect free.
            eval_client=app.registry.get('evaluator',evaluator['framework']['adapter'])
            eval_check={'status':'not_supported'}
            try:
                if 'plan_preflight' in (eval_client.identity.manifest.get('operations') or []):
                    # The evaluator may need two bounded Git probes on a shared
                    # filesystem.  Keep Core's outer RPC budget above those
                    # internal limits so doctor does not manufacture a timeout.
                    pf=eval_client.invoke('plan_preflight',{'evaluation':evaluator,'cache_root':bundle.cache_root},context={'doctor':True,'cache_root':bundle.cache_root,'offline':True,'preflight':True},timeout=EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS)
                    proc=copy.deepcopy(pf['process']); proc.setdefault('metadata',{})['role']='doctor_evaluator_preflight'
                    proc,warning_events=wrapped_process(proc,platform_spec=platform,resolved_platform=resolved_platform,role='evaluator')
                    cp=orch.pm.run(proc)
                    eval_check={'status':'ok' if cp.returncode==0 else 'failed','returncode':cp.returncode,'stdout':text(cp.stdout)[:4000],'stderr':text(cp.stderr)[:4000]}
                    if pf.get('result_format')=='preflight_result':
                        try:
                            result=orch._preflight_json_result(text(cp.stdout)); app.schemas.validate('preflight_probe_result',result); eval_check['result']=result
                            if (cp.returncode==0) != (result['status']=='passed'):
                                eval_check['status']='failed'; eval_check['error']='preflight process/result status mismatch'
                        except Exception as exc:
                            eval_check['status']='failed'; eval_check['error']=f'{type(exc).__name__}: {exc}'
                    row['warnings'].extend(warning_events)
                else:
                    snap=eval_client.invoke('snapshot',{'evaluation':evaluator},context={'doctor':True,'offline':True},timeout=EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS)
                    eval_check={'status':'ok' if not snap.get('probe_error') else 'failed','snapshot':snap}
            except Exception as exc:
                eval_check={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            row['checks']['evaluator_environment']=eval_check

            # Managed backend probe: plan only, then execute the adapter's
            # declared version command inside the selected backend environment
            # with the same device/runtime/backend EnvPatch layers used by run.
            backend_check={'status':'deferred_external'}
            if mode=='managed':
                backend=app.registry.get('backend',deployment['backend']['adapter'])
                try:
                    offline=bool((child['run_spec'].get('overrides') or {}).get('offline',False))
                    start_input={
                        'model':model,'deployment':deployment,'platform':resolved_platform,
                        'endpoint':child['resolved'].get('endpoint',{}),
                        'log_path':str(Path(bundle.cache_root)/'doctor-backend.log'),
                        'network_policy':'offline' if offline else 'online',
                    }
                    if 'plan_preflight' in (backend.identity.manifest.get('operations') or []):
                        preflight_input={
                            'model':model,'deployment':deployment,'platform':resolved_platform,
                            'network_policy':'offline' if offline else 'online',
                        }
                        preflight=backend.invoke('plan_preflight',preflight_input,context={'doctor':True,'offline':offline,'preflight':True},timeout=5)
                        report=orch.run_backend_preflight(
                            preflight,platform_spec=platform,resolved_platform=resolved_platform,
                            raise_on_failure=False,
                        )
                        backend_check={
                            'status':'ok' if report['status']=='passed' else 'failed',
                            'preflight':report,
                        }
                    else:
                        start_plan=backend.invoke('plan_start',start_input,context={'doctor':True,'offline':offline},timeout=5)
                        probe=copy.deepcopy(start_plan['dependency_probe'])
                        probe_argv=list(probe.get('argv') or [])
                        probe.setdefault('metadata',{})['role']='doctor_backend_dependency_probe'
                        probe,warning_events=wrapped_process(
                            probe,platform_spec=platform,resolved_platform=resolved_platform,role='backend',
                            base_patches=(("device",resolved_platform.get('device_env_patch')),("runtime",resolved_platform.get('runtime_env_patch'))),
                        )
                        cp=orch.pm.run(probe)
                        backend_check={'status':'ok' if cp.returncode==0 else 'failed','returncode':cp.returncode,'argv':probe_argv,'wrapped_argv':probe.get('argv'),'stdout':text(cp.stdout)[:4000],'stderr':text(cp.stderr)[:4000]}
                        row['warnings'].extend(warning_events)
                except Exception as exc:
                    backend_check={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            row['checks']['backend_environment']=backend_check
            rows.append(row)

        failed_statuses={'failed'}
        local_ok=all(
            r['compatibility']!='incompatible' and
            all(c.get('status') not in failed_statuses for c in r['checks'].values())
            for r in rows
        )
        doctor_payload={
            'ok':local_ok,'scope':'local-workload-preflight-no-service-start',
            'system':bundle.system['system']['name'],'selected_profiles':bundle.generated.get('selected_profiles',{}),
            'runs':rows,
        }
        doctor_dump(doctor_payload, orch)
        raise SystemExit(0 if local_ok else 2)
    if args.cmd=='plan':
        if args.run:
            plan=app.plan(args.run)
        else:
            plan,_bundle=app.user_matrix_plan(args.system_config,args.evaluation_config)
        if args.output: atomic_json(args.output,plan)
        dump(plan); return
    if args.cmd=='run':
        if args.run:
            plan=app.plan(args.run); orch=app.orchestrator(results_root=args.results_root,cache_root=args.cache_root); path=orch.execute(plan); print(path); return
        plan,bundle=app.user_matrix_plan(args.system_config,args.evaluation_config)
        exe=app.matrix_executor(results_root=args.results_root or bundle.results_root,cache_root=args.cache_root or bundle.cache_root)
        path,summary=exe.execute(plan); dump({'batch_dir':str(path),'summary':summary}); raise SystemExit(3 if summary['failed'] or summary['not_run'] else 0)
    if args.cmd=='run-plan':
        raw=json_loads_strict(Path(args.plan).read_text(encoding='utf-8'))
        if isinstance(raw,dict) and 'matrix_id' in raw:
            plan=app.load_matrix_plan(args.plan); user_paths=(plan.get('summary') or {}).get('user_config') or {}
            exe=app.matrix_executor(results_root=args.results_root or user_paths.get('results_root'),cache_root=args.cache_root or user_paths.get('cache_root')); path,summary=exe.execute(plan,continue_on_error=args.continue_on_error,resume_dir=args.resume_dir); dump({'batch_dir':str(path),'summary':summary}); raise SystemExit(3 if summary['failed'] or summary['not_run'] else 0)
        plan=app.load_plan(args.plan); orch=app.orchestrator(results_root=args.results_root,cache_root=args.cache_root); path=orch.execute(plan); print(path); return
    if args.cmd=='matrix-validate':
        spec=app.matrices.load(args.matrix); dump({'ok':True,'matrix':spec}); return
    if args.cmd=='matrix-expand':
        runs=app.matrix_expand(args.matrix); obj={'runs':runs,'count':len(runs)}
        if args.output: atomic_json(args.output,obj)
        dump(obj); return
    if args.cmd=='matrix-plan':
        plan=app.matrix_plan(args.matrix)
        if args.output: atomic_json(args.output,plan)
        dump(plan); return
    if args.cmd=='matrix-run':
        plan=app.matrix_plan(args.matrix); exe=app.matrix_executor(results_root=args.results_root,cache_root=args.cache_root); path,summary=exe.execute(plan,continue_on_error=args.continue_on_error); dump({'batch_dir':str(path),'summary':summary}); raise SystemExit(3 if summary['failed'] or summary['not_run'] else 0)
    if args.cmd=='run-matrix-plan':
        plan=app.load_matrix_plan(args.plan); user_paths=(plan.get('summary') or {}).get('user_config') or {}
        exe=app.matrix_executor(results_root=args.results_root or user_paths.get('results_root'),cache_root=args.cache_root or user_paths.get('cache_root')); path,summary=exe.execute(plan,continue_on_error=args.continue_on_error,resume_dir=args.resume_dir); dump({'batch_dir':str(path),'summary':summary}); raise SystemExit(3 if summary['failed'] or summary['not_run'] else 0)
def main():
    try: _main()
    except ModelEvalError as exc:
        print(f'{getattr(exc,"code","MODEL_EVAL_ERROR")}: {exc}',file=sys.stderr)
        details=getattr(exc,'details',{}) or {}
        if details.get('run_dir'): print(f'run_dir: {details["run_dir"]}',file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':
    main()
