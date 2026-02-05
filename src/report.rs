//! Report generation for audit results

use anyhow::Result;
use minijinja::{context, Environment};
use std::fs;
use std::path::Path;

use crate::models::AuditResult;

/// Trait for report generators
pub trait Reporter {
    fn generate(&self, result: &AuditResult, output_path: &Path) -> Result<()>;
}

/// JSON report generator
pub struct JsonReporter;

impl JsonReporter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Reporter for JsonReporter {
    fn generate(&self, result: &AuditResult, output_path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(result)?;
        fs::write(output_path, json)?;
        Ok(())
    }
}

/// HTML report generator
pub struct HtmlReporter;

impl HtmlReporter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HtmlReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Reporter for HtmlReporter {
    fn generate(&self, result: &AuditResult, output_path: &Path) -> Result<()> {
        let mut env = Environment::new();
        env.add_template("report", HTML_TEMPLATE)?;

        let template = env.get_template("report")?;
        let html = template.render(context! {
            result => result,
            generated_at => &result.generated_at,
            dataset_path => &result.dataset_path,
            method => &result.method,
            confidence_threshold => result.confidence_threshold,
            iou_threshold => result.iou_threshold,
            total_images => result.total_images,
            images_audited => result.images_audited,
            images_with_issues => result.images_with_issues(),
            total_issues => result.total_issues(),
            high_count => result.high_count(),
            medium_count => result.medium_count(),
            low_count => result.low_count(),
            issues_by_type => result.issues_by_type(),
            flagged_images => &result.flagged_images,
        })?;

        fs::write(output_path, html)?;
        Ok(())
    }
}

const HTML_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GT Audit Report</title>
    <style>
        :root {
            --primary: #00d4ff;
            --bg-dark: #0a0a12;
            --bg-card: rgba(255, 255, 255, 0.03);
            --border: rgba(255, 255, 255, 0.08);
            --text: #e4e4e4;
            --text-muted: #888;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; color: var(--primary); }
        .meta { color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .summary-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        .summary-card .value {
            font-size: 2.5rem;
            font-weight: 700;
        }
        .summary-card .label {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        .high { color: var(--error); }
        .medium { color: var(--warning); }
        .low { color: var(--text-muted); }
        .issues-section { margin-top: 2rem; }
        .issues-section h2 {
            color: var(--primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        .issue-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        .issue-card.high { border-left: 4px solid var(--error); }
        .issue-card.medium { border-left: 4px solid var(--warning); }
        .issue-card.low { border-left: 4px solid var(--text-muted); }
        .issue-header {
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .issue-header:hover { background: rgba(0,0,0,0.3); }
        .issue-filename { font-family: monospace; }
        .badge {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        .badge-high { background: var(--error); color: white; }
        .badge-medium { background: var(--warning); color: black; }
        .badge-low { background: var(--text-muted); color: white; }
        .issue-details {
            padding: 1rem;
            display: none;
        }
        .issue-details.expanded { display: block; }
        .issue-item {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
            font-size: 0.85rem;
        }
        .issue-type { color: var(--primary); font-weight: 600; }
        footer {
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        footer a { color: var(--primary); }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ground Truth Audit Report</h1>
        <p class="meta">
            Generated: {{ generated_at }} | Method: {{ method }} |
            Confidence: {{ confidence_threshold }} | IoU: {{ iou_threshold }}
        </p>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{{ total_images }}</div>
                <div class="label">Total Images</div>
            </div>
            <div class="summary-card">
                <div class="value">{{ images_audited }}</div>
                <div class="label">Images Audited</div>
            </div>
            <div class="summary-card">
                <div class="value">{{ images_with_issues }}</div>
                <div class="label">With Issues</div>
            </div>
            <div class="summary-card">
                <div class="value high">{{ high_count }}</div>
                <div class="label">High Severity</div>
            </div>
            <div class="summary-card">
                <div class="value medium">{{ medium_count }}</div>
                <div class="label">Medium Severity</div>
            </div>
            <div class="summary-card">
                <div class="value low">{{ low_count }}</div>
                <div class="label">Low Severity</div>
            </div>
        </div>

        <div class="issues-section">
            <h2>Flagged Images ({{ flagged_images|length }})</h2>
            {% for img in flagged_images %}
            <div class="issue-card {% if img.high_count() > 0 %}high{% elif img.medium_count() > 0 %}medium{% else %}low{% endif %}">
                <div class="issue-header" onclick="toggleDetails(this)">
                    <span class="issue-filename">{{ img.filename }}</span>
                    <span>
                        {% if img.high_count() > 0 %}<span class="badge badge-high">{{ img.high_count() }} HIGH</span>{% endif %}
                        {% if img.medium_count() > 0 %}<span class="badge badge-medium">{{ img.medium_count() }} MED</span>{% endif %}
                        <span class="badge badge-low">{{ img.issues|length }} total</span>
                    </span>
                </div>
                <div class="issue-details">
                    <p style="color: var(--text-muted); margin-bottom: 0.5rem; font-size: 0.8rem;">
                        GT: {{ img.gt_count }} objects | Detected: {{ img.detection_count }}
                    </p>
                    {% for issue in img.issues %}
                    <div class="issue-item">
                        <span class="issue-type">{{ issue.issue_type }}</span>: {{ issue.description }}
                        {% if issue.explanation %}<br><small style="color: var(--text-muted);">{{ issue.explanation }}</small>{% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>

        <footer>
            <p>Generated by <a href="https://github.com/ARTIFACTIQ/gt-audit">gt-audit</a> |
               <a href="https://artifactiq.ai">Artifactiq</a></p>
        </footer>
    </div>

    <script>
        function toggleDetails(header) {
            const details = header.nextElementSibling;
            details.classList.toggle('expanded');
        }
    </script>
</body>
</html>
"#;
