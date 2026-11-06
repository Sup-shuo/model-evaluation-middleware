# MMStar with EvalScope

MMStar evaluates image-and-text inputs. vLLM serves the checkpoint through the OpenAI-compatible Chat Completions API; EvalScope prepares the validation split, sends the images and scores responses. The middleware selects the checkpoint, probes the image endpoint and records the result.

1. Copy the [example](../examples/mmstar). In `system.yaml`, set the inference executable, hardware, model root and a Python environment with EvalScope. The example uses CUDA; use the hardware and runtime profiles for your machine.
2. Replace the illustrative model references with directories you actually have. The selected model needs a vision encoder, any required projector, its processor files, and a backend that supports that architecture and weight format. Allow enough context for the image, prompt and response.
3. From the project root, validate the selection and run:

```bash
./eval-manager validate --system-config examples/mmstar/system.yaml --evaluation-config examples/mmstar/evaluation.yaml
./eval-manager evaluate --system-config examples/mmstar/system.yaml --evaluation-config examples/mmstar/evaluation.yaml --smoke
./eval-manager evaluate --system-config examples/mmstar/system.yaml --evaluation-config examples/mmstar/evaluation.yaml
```

For a native multimodal checkpoint, specify its catalog ID directly. To reuse an original ID while selecting a separate visual checkpoint, opt in with `select_artifact_by_modality: true` and map `image_text` to that checkpoint's catalog ID. This selects an existing artifact; it never converts or downloads weights. Set `language_model_only: false` in the evaluation when a backend profile or model default was intended for text-only tasks.

For a full result, the MMStar preset checks the validation split (six categories of 250), image-bearing predictions and matching reviews. A request error or context overflow invalidates the result; a smoke run is for integration, not a benchmark score. Older archived results are not retroactively validated.

Manual Qwen conversions store vision and MTP (multi-token prediction) retention separately in `DERIVATION.json`. A missing field in a legacy artifact is unknown. A preserved vision parameter set still requires a working image processor and service.

## 中文速查

使用已有的完整图文权重时，直接配置它的模型 ID。如果同一模型有文本版和保留视觉的派生版，可以在模型目录声明 `metadata.artifacts_by_modality.image_text`，并在本次评测为该模型启用 `select_artifact_by_modality`。先替换示例中的路径，再运行上面的命令。纯语言派生不能用于 MMStar；MTP 是否保留与能否接收图片是两件事。验收看全量样本、图片和评分记录，不只看有没有总分。
