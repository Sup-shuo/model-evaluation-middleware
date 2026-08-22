# Configuration management

The configuration catalog can grow without adding YAML inheritance or hidden
defaults. System and Evaluation references support confined subdirectories;
Model identity remains the globally unique `id` declared inside each Model
file.

## List and display

```bash
eval-manager config list
eval-manager config list --kind model --format json
eval-manager config show system lab/nvidia
eval-manager config show evaluation teams/bbh
eval-manager config show model qwen35-08b-base
```

`show` displays the stored document. Use `eval-manager plan` when you need the
fully resolved execution plan and effective overrides.

## Check the complete catalog

```bash
eval-manager config check
eval-manager config check --kind evaluation
eval-manager config check \
  --system-config lab/nvidia \
  --evaluation-config teams/bbh
```

Catalog checking reports malformed YAML, unsupported Schema versions,
duplicate global Model IDs, duplicate System/Evaluation references, and Schema
violations. Supplying both configuration options additionally validates their
cross-references and selected profiles. It does not start a Backend.

## Migrate public user Schemas

Migration is preview-only unless `--write` is supplied:

```bash
eval-manager config migrate
eval-manager config migrate --write
```

The 1.2 to 1.3 Evaluation migration moves Backend and Evaluator parameters
under explicit profile selections. If the old Evaluation did not select those
profiles, the tool refuses to guess:

```bash
eval-manager config migrate \
  --kind evaluation \
  --backend-profile vllm \
  --evaluator-profile lm_eval

# Review first, then repeat with --write.
```

Writing preflights the complete selected catalog, uses atomic replacement for
each document, and rolls back earlier replacements if a later write fails.
Migrated YAML is normalized, so comments and custom formatting are not retained.
It does not alter model material or touch results. Keep configuration under
normal version control and review the dry-run before execution.

## Deliberate limits

- The catalog does not implement YAML `extends`, includes, or inheritance.
- Directories organize System and Evaluation references, but Model directories
  never change the Model ID.
- Migration only handles explicitly implemented version transitions; unknown
  versions fail closed.
- Catalog checks validate configuration, not real hardware availability. Use
  `check` or `doctor` for a selected machine and Evaluation.
