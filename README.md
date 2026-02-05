# gt-audit

Automated ground truth label validation for object detection datasets.

**The problem:** Bad labels silently corrupt your ML metrics. Manual review doesn't scale.

**The solution:** Use AI models to audit your labels. When the model disagrees with your ground truth, flag it for review.

## Features

- **Zero-shot detection** (CPU) - Grounding DINO + CLIP for robust validation without training
- **Open VLM** (GPU) - LLaVA/InternVL for semantic understanding when GPU available
- **YOLO-native** - Works directly with YOLO format labels (class x y w h)
- **CI/CD ready** - Exit codes, thresholds, JSON output for pipelines
- **Standalone reports** - Single HTML file, no server needed

## Installation

```bash
pip install gt-audit
```

Or install from source:

```bash
git clone https://github.com/ARTIFACTIQ/gt-audit.git
cd gt-audit
pip install -e .
```

## Quick Start

```bash
# Validate a YOLO dataset (auto-selects best available method)
gt-audit validate ./my-dataset

# Force CPU mode (zero-shot detection)
gt-audit validate ./my-dataset --method zero-shot

# Use VLM for deeper semantic analysis (requires GPU)
gt-audit validate ./my-dataset --method vlm

# Generate HTML report
gt-audit validate ./my-dataset --output report.html

# CI mode: fail if too many issues
gt-audit validate ./my-dataset --fail-on-high 5 --fail-on-medium 20
```

## Dataset Structure

gt-audit expects YOLO format:

```
my-dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── image3.jpg
├── labels/
│   ├── train/
│   │   ├── image1.txt    # class x y w h (normalized)
│   │   └── image2.txt
│   └── val/
│       └── image3.txt
└── dataset.yaml          # class names
```

Or flat structure:

```
my-dataset/
├── images/
│   └── *.jpg
├── labels/
│   └── *.txt
└── classes.txt           # one class name per line
```

## Detection Methods

### Zero-Shot (CPU) - Default

Uses Grounding DINO for detection + CLIP for class verification:

```bash
gt-audit validate ./dataset --method zero-shot
```

**How it works:**
1. Grounding DINO detects objects using class names as text prompts
2. CLIP verifies detected regions match the expected class
3. Compares to GT labels, flags mismatches

**Pros:** No training needed, works on CPU, robust
**Cons:** Slower than trained models, may miss domain-specific classes

### Open VLM (GPU)

Uses LLaVA or InternVL for semantic understanding:

```bash
gt-audit validate ./dataset --method vlm --model llava-1.6
```

**How it works:**
1. For each labeled region, asks VLM: "Is this a {class}?"
2. VLM provides yes/no + explanation
3. Flags disagreements with reasoning

**Pros:** Best semantic understanding, explains why labels are wrong
**Cons:** Requires GPU, slower, higher memory

### BYOM (Bring Your Own Model)

Use your own trained model:

```bash
gt-audit validate ./dataset --method byom --model ./my-model.pt
```

## Issue Types

| Type | Severity | Description |
|------|----------|-------------|
| `class_mismatch` | High | Model detects different class than GT label |
| `missing_label` | Medium | Model detects object with no GT label |
| `spurious_label` | Low | GT label exists but model detects nothing |
| `localization` | Low | Box position/size significantly different |

## Output Formats

### JSON (default)

```bash
gt-audit validate ./dataset --format json --output audit.json
```

```json
{
  "summary": {
    "total_images": 1000,
    "images_with_issues": 45,
    "issues_by_severity": {"high": 7, "medium": 23, "low": 156}
  },
  "issues": [
    {
      "image": "image1.jpg",
      "severity": "high",
      "type": "class_mismatch",
      "gt_class": "Boot",
      "detected_class": "Dress",
      "confidence": 0.92,
      "explanation": "Image shows a floral dress, not footwear"
    }
  ]
}
```

### HTML Report

```bash
gt-audit validate ./dataset --format html --output report.html
```

Generates a standalone HTML file with:
- Summary statistics
- Filterable issue list
- Image previews
- Click to expand details

### CI Mode

```bash
gt-audit validate ./dataset \
  --fail-on-high 5 \
  --fail-on-medium 20 \
  --format json \
  --output audit.json

echo $?  # Exit code: 0 = pass, 1 = fail
```

## Configuration

Create `gt-audit.yaml` in your dataset directory:

```yaml
# Detection settings
method: auto              # auto, zero-shot, vlm, byom
confidence_threshold: 0.3
iou_threshold: 0.5

# Class equivalences (don't flag these as mismatches)
class_groups:
  - [Clothing, Dress, Suit, Jacket, Coat]
  - [Footwear, Boot, Sandal, High heels]
  - [Bag, Handbag, Backpack, Briefcase]

# Sampling (for large datasets)
sample_size: 1000         # 0 = all images
sample_seed: 42

# CI thresholds
fail_on_high: 10
fail_on_medium: 50
```

## Python API

```python
from gt_audit import Auditor

# Initialize with auto-detected method
auditor = Auditor(method="auto")

# Validate dataset
results = auditor.validate("./my-dataset")

# Access results
print(f"Issues found: {results.total_issues}")
print(f"High severity: {results.high_count}")

# Filter issues
for issue in results.filter(severity="high"):
    print(f"{issue.image}: {issue.gt_class} -> {issue.detected_class}")

# Generate report
results.to_html("report.html")
results.to_json("audit.json")
```

## GitHub Actions

```yaml
name: Validate Ground Truth

on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install gt-audit
        run: pip install gt-audit

      - name: Run audit
        run: |
          gt-audit validate ./data \
            --fail-on-high 5 \
            --format json \
            --output audit.json

      - name: Upload report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: gt-audit-report
          path: audit.json
```

## Requirements

- Python 3.9+
- For zero-shot: ~4GB disk (model downloads)
- For VLM: NVIDIA GPU with 8GB+ VRAM

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## Related Projects

- [ml-gpu](https://github.com/ARTIFACTIQ/mlgpu) - GPU setup for ML training
- [cleanlab](https://github.com/cleanlab/cleanlab) - General data quality
- [FiftyOne](https://github.com/voxel51/fiftyone) - Dataset visualization

---

Built by [Artifactiq](https://artifactiq.ai) - Committed to transparency in AI.
