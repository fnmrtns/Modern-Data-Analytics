import sys
from pathlib import Path
import yaml
import importlib.util
import inspect
import pandas as pd

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

# Add 'src' to Python path so Kedro can find 'startupdelay_horizon'
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from startupdelay_horizon.pipeline_registry import register_pipelines

# Configure the Kedro project
configure_project("startupdelay_horizon")

# Start Kedro session and get context and pipelines
project_path = Path.cwd()
with KedroSession.create(project_path=project_path, env="local") as session:
    context = session.load_context()
    pipelines = register_pipelines()

# Load catalog
catalog_path = Path("conf/base/catalog.yml")
with open(catalog_path, "r") as f:
    catalog = yaml.safe_load(f)

# Start writing report
lines = ["# 📘 Kedro Project Report\n"]

# Pipelines
lines.append("## 📊 Pipelines\n")
for name, pipeline in pipelines.items():
    lines.append(f"### `{name}`\n")
    for node in pipeline.nodes:
        inputs = node.inputs if isinstance(node.inputs, (list, tuple)) else [node.inputs]
        outputs = node.outputs if isinstance(node.outputs, (list, tuple)) else [node.outputs]
        lines.append(f"- **Node:** `{node.name}`")
        lines.append(f"  - 📥 Inputs: {', '.join(inputs)}")
        lines.append(f"  - 📤 Outputs: {', '.join(outputs)}")
        lines.append(f"  - 🧠 Function: `{node.func.__name__}`")
        lines.append("")

# Catalog
lines.append("\n## 📁 Data Catalog\n")
for dataset_name, dataset_meta in catalog.items():
    dtype = dataset_meta.get("type", "Unknown")
    fpath = dataset_meta.get("filepath", "—")
    lines.append(f"- `{dataset_name}`: **{dtype}** → `{fpath}`")

# Node Source Code (optional)
lines.append("\n## 🧠 Node Function Code (Top-Level Only)\n")
functions_seen = set()
for pipeline in pipelines.values():
    for node in pipeline.nodes:
        func = node.func
        if func not in functions_seen:
            functions_seen.add(func)
            try:
                code = inspect.getsource(func)
                lines.append(f"### `{func.__name__}`")
                lines.append("```python")
                lines.append(code.strip())
                lines.append("```\n")
            except Exception:
                lines.append(f"### `{func.__name__}` — _source not available_\n")

# Write the report to Markdown
with open("kedro_project_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ Report generated: kedro_project_report.md")
