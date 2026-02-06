# gt-audit

Fast ground truth label validation for object detection datasets. Written in Rust.

**The problem:** Bad labels silently corrupt your ML metrics. Manual review doesn't scale.

**The solution:** Use your trained model to audit labels. When the model disagrees with ground truth, flag it for review.

## Features

- **Fast** - Rust + ONNX Runtime, processes 1000s of images per minute
- **YOLO-native** - Works directly with YOLO format labels (class x y w h)
- **Bring Your Own Model** - Use any ONNX model for validation
- **CI/CD ready** - JSON output, exit codes for pipelines
- **Single binary** - No Python, no dependencies

## Installation

### Quick Install (Linux/macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/ARTIFACTIQ/gt-audit/main/install.sh | bash
```

### From Releases

Download from [GitHub Releases](https://github.com/ARTIFACTIQ/gt-audit/releases):

```bash
# Linux
curl -LO https://github.com/ARTIFACTIQ/gt-audit/releases/latest/download/gt-audit-linux-x86_64.tar.gz
tar -xzf gt-audit-linux-x86_64.tar.gz
sudo mv gt-audit /usr/local/bin/

# macOS (Apple Silicon)
curl -LO https://github.com/ARTIFACTIQ/gt-audit/releases/latest/download/gt-audit-darwin-arm64.tar.gz
tar -xzf gt-audit-darwin-arm64.tar.gz
sudo mv gt-audit /usr/local/bin/
```

### Build from Source

```bash
git clone https://github.com/ARTIFACTIQ/gt-audit.git
cd gt-audit
cargo build --release
```

## Quick Start

```bash
# Validate dataset using your trained ONNX model
gt-audit validate ./my-dataset --model ./model.onnx

# Adjust confidence threshold
gt-audit validate ./my-dataset --model ./model.onnx --confidence 0.3

# Sample subset for quick check
gt-audit validate ./my-dataset --model ./model.onnx --sample 100

# Output to JSON file
gt-audit validate ./my-dataset --model ./model.onnx --output audit.json

# Generate HTML report
gt-audit validate ./my-dataset --model ./model.onnx --output report.html
```

## Dataset Structure

gt-audit expects YOLO format:

```
my-dataset/
├── images/
│   └── val/
│       ├── image1.jpg
│       └── image2.jpg
├── labels/
│   └── val/
│       ├── image1.txt    # class x y w h (normalized)
│       └── image2.txt
└── dataset.yaml          # with 'names:' listing class names
```

Or with `classes.txt`:

```
my-dataset/
├── images/val/*.jpg
├── labels/val/*.txt
└── classes.txt           # one class name per line
```

## Issue Types

| Type | Severity | Description |
|------|----------|-------------|
| `class_mismatch` | High | Model detects different class than GT label |
| `missing_label` | Medium | Model detects object with no GT label nearby |
| `spurious_label` | Low | GT label exists but model detects nothing there |

## Output Format

### JSON

```json
{
  "generator": "gt-audit",
  "generator_version": "0.2.0",
  "generated_at": "2026-02-06T05:26:16Z",
  "method": "yolo",
  "confidence_threshold": 0.25,
  "summary": {
    "total_images": 700,
    "images_audited": 700,
    "images_with_issues": 642,
    "total_issues": 1905,
    "by_severity": {"high": 148, "medium": 68, "low": 1689},
    "by_type": {"class_mismatch": 148, "missing_label": 68, "spurious_label": 1689}
  },
  "flagged_images": [...]
}
```

### Terminal Output

```
╔══════════════════════════════════════════════════════════╗
║               gt-audit - Ground Truth Validator          ║
╚══════════════════════════════════════════════════════════╝

📂 Loading dataset: ./my-dataset
   Classes: 39
   Images: 700

🔍 Initializing detector: yolo
   Loading ONNX model: ./model.onnx

🔬 Auditing 700 images...

╔══════════════════════════════════════════════════════════╗
║                      AUDIT SUMMARY                       ║
╚══════════════════════════════════════════════════════════╝

  Total images:       700
  Images audited:     700
  Images with issues: 642
  Total issues:       1905

  By severity:
    🔴 High:   148
    🟡 Medium: 68
    ⚪ Low:    1689

  Time: 45.2s
```

## GitHub Actions

```yaml
name: GT Audit

on:
  release:
    types: [published]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Download gt-audit
        run: |
          curl -fsSL https://raw.githubusercontent.com/ARTIFACTIQ/gt-audit/main/install.sh | bash
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Export model to ONNX
        run: |
          pip install ultralytics torch --index-url https://download.pytorch.org/whl/cpu
          python -c "from ultralytics import YOLO; YOLO('model.pt').export(format='onnx')"

      - name: Run audit
        run: gt-audit validate . --model model.onnx --output audit.json

      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: gt-audit-report
          path: audit.json
```

## CLI Reference

```
gt-audit validate <DATASET> [OPTIONS]

Arguments:
  <DATASET>  Path to dataset (YOLO format)

Options:
  -m, --model <PATH>       Path to ONNX model
  -c, --confidence <FLOAT> Confidence threshold [default: 0.25]
      --iou <FLOAT>        IoU threshold for matching [default: 0.5]
  -o, --output <PATH>      Output file (json or html based on extension)
      --sample <N>         Sample N images (0 = all) [default: 0]
      --seed <N>           Random seed for sampling [default: 42]
  -h, --help               Print help
  -V, --version            Print version
```

## Requirements

- Linux x86_64 or macOS ARM64
- ONNX model exported from YOLOv8/v11

## License

MIT License - see [LICENSE](LICENSE)

---

Built by [Artifactiq](https://artifactiq.ai) - Committed to transparency in AI.
