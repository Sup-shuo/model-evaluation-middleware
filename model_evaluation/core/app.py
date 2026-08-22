from __future__ import annotations
from pathlib import Path
import os
import tempfile
from model_evaluation.core.config.loader import SpecRepository
from model_evaluation.core.planner import Planner
from model_evaluation.core.process.manager import ProcessManager, SecretStore
from model_evaluation.core.registry.adapter_registry import AdapterRegistry
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.orchestrator import Orchestrator
from model_evaluation.core.matrix import MatrixSchemas, MatrixRepository, MatrixPlanner, MatrixExecutor, verify_matrix_plan, finalize_matrix_plan
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.core.user_config import UserConfigResolver
from model_evaluation.core.errors import ConfigError
from model_evaluation.core.execution_plan import validate_execution_plan
from model_evaluation.core.matrix_export import export_execution_plans
from model_evaluation.core.config.catalog import resolve_config_reference


def _host_runtime_root() -> Path:
    explicit=os.environ.get("MODEL_EVAL_RUNTIME_ROOT")
    if explicit:
        root=Path(explicit).expanduser()
        if not root.is_absolute():
            raise ValueError("MODEL_EVAL_RUNTIME_ROOT must be absolute")
        return root.resolve()
    xdg=os.environ.get("XDG_RUNTIME_DIR")
    if xdg and Path(xdg).is_absolute():
        return (Path(xdg)/"model-evaluation-middleware").resolve()
    return (Path(tempfile.gettempdir())/f"model-evaluation-middleware-{os.getuid()}").resolve()

class Application:
    def __init__(self, package_root: str | Path, project_root: str | Path | None = None):
        # ``root`` owns packaged schemas/adapters/presets. ``project_root`` owns
        # mutable user configuration and run products.
        self.root=Path(package_root).resolve()
        self.project_root=Path(project_root if project_root is not None else package_root).resolve()
        self.host_runtime_root=_host_runtime_root()
        self.schemas=SchemaStore(self.root/'schemas')
        self.specs=SpecRepository(self.root/'presets',self.schemas)
        self.registry=AdapterRegistry(self.root/'adapters',self.schemas)
        self.planner=Planner(project_root=self.root,schemas=self.schemas,specs=self.specs,registry=self.registry)
        self.matrix_schemas=MatrixSchemas(self.root/'schemas'/'user')
        self.matrices=MatrixRepository(self.root/'presets'/'matrices',self.matrix_schemas)
        self.matrix_planner=MatrixPlanner(self)
        self.user_config=UserConfigResolver(self)

    def plan(self, run_or_path: str) -> dict:
        return self.planner.build(self.specs.resolve_run(run_or_path))

    def _user_config_path(self, value: str | Path | None, *, env_name: str, legacy_name: str, catalog_dir: str) -> Path:
        raw_value=value or os.environ.get(env_name)
        if raw_value is None:
            return self.project_root/'config'/legacy_name
        return resolve_config_reference(
            self.project_root,
            raw_value,
            catalog_dir=catalog_dir,
        )

    def load_user_config(
        self,
        system_path: str | Path | None = None,
        evaluation_path: str | Path | None = None,
        *,
        smoke: bool = False,
    ):
        # Explicit CLI/API paths win.  Otherwise each machine may pin its own
        # system configuration through the process environment while reusing
        # the same evaluation file across machines.
        system_value = self._user_config_path(
            system_path, env_name="MODEL_EVAL_SYSTEM_CONFIG", legacy_name="system.yaml", catalog_dir="systems"
        )
        evaluation_value = self._user_config_path(
            evaluation_path, env_name="MODEL_EVAL_EVALUATION_CONFIG", legacy_name="evaluation.yaml", catalog_dir="evaluations"
        )
        return self.user_config.load(
            Path(system_value).expanduser(),
            Path(evaluation_value).expanduser(),
            smoke=smoke,
        )

    def user_matrix_plan(
        self,
        system_path: str | Path | None = None,
        evaluation_path: str | Path | None = None,
        *,
        smoke: bool = False,
    ) -> tuple[dict, object]:
        bundle = self.load_user_config(
            system_path,
            evaluation_path,
            smoke=smoke,
        )
        return self.build_user_matrix_plan(bundle), bundle

    def build_user_matrix_plan(self, bundle) -> dict:
        """Build a user matrix from the bundle's detached spec snapshot.

        Keeping this operation explicit lets API callers load more than one
        configuration on the same Application and plan either bundle later;
        the shared ``app.specs`` compatibility view is not consulted.
        """
        plan=self.matrix_planner.build(bundle.matrix_spec,specs=bundle.specs)
        plan.setdefault('summary',{})['user_config']={
            'cache_root': bundle.cache_root,
            'results_root': bundle.results_root,
            'system_name': bundle.system['system']['name'],
            'selected_profiles': bundle.generated.get('selected_profiles', {}),
        }
        finalize_matrix_plan(plan)
        self.matrix_schemas.validate('matrix_plan',plan)
        return plan

    def load_plan(self, path: str | Path) -> dict:
        obj=json_loads_strict(Path(path).read_text(encoding='utf-8'))
        validate_execution_plan(obj,self.schemas)
        return obj


    def matrix_expand(self, matrix_or_path: str) -> list[dict]:
        return self.matrix_planner.expand(self.matrices.load(matrix_or_path))

    def matrix_plan(self, matrix_or_path: str) -> dict:
        return self.matrix_planner.build(self.matrices.load(matrix_or_path))

    def load_matrix_plan(self, path: str | Path) -> dict:
        obj=json_loads_strict(Path(path).read_text(encoding='utf-8'))
        verify_matrix_plan(obj,app=self)
        return obj

    def export_matrix_plan(
        self,
        plan: dict,
        output_dir: str | Path,
        *,
        shards: int,
        strategy: str = "round_robin",
    ) -> dict:
        verify_matrix_plan(plan, app=self)
        return export_execution_plans(
            plan,
            output_dir,
            shards=shards,
            schemas=self.matrix_schemas,
            strategy=strategy,
        )

    def matrix_executor(self, *, results_root: str | Path | None=None, cache_root: str | Path | None=None, secrets: dict[str,str] | None=None):
        return MatrixExecutor(self,results_root=results_root,cache_root=cache_root,secrets_map=secrets)
    def orchestrator(self, *, results_root: str | Path | None=None, cache_root: str | Path | None=None, secrets: dict[str,str] | None=None):
        pm=ProcessManager(self.schemas,secrets=SecretStore(secrets),ownership_root=self.host_runtime_root/'processes')
        rm=ResourceManager(self.host_runtime_root/'resources')
        return Orchestrator(project_root=self.root,schemas=self.schemas,registry=self.registry,process_manager=pm,resource_manager=rm,results_root=results_root or self.project_root/'results',cache_root=cache_root or self.project_root/'cache')
