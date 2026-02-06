//! Data models for gt-audit

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    ClassMismatch,
    MissingLabel,
    SpuriousLabel,
    Localization,
}

impl std::fmt::Display for IssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueType::ClassMismatch => write!(f, "class_mismatch"),
            IssueType::MissingLabel => write!(f, "missing_label"),
            IssueType::SpuriousLabel => write!(f, "spurious_label"),
            IssueType::Localization => write!(f, "localization"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BoundingBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Convert to x1, y1, x2, y2 format (normalized)
    pub fn to_xyxy(&self) -> (f32, f32, f32, f32) {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        (x1, y1, x2, y2)
    }

    /// Calculate IoU with another bounding box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let (ax1, ay1, ax2, ay2) = self.to_xyxy();
        let (bx1, by1, bx2, by2) = other.to_xyxy();

        let x1 = ax1.max(bx1);
        let y1 = ay1.max(by1);
        let x2 = ax2.min(bx2);
        let y2 = ay2.min(by2);

        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area_a = (ax2 - ax1) * (ay2 - ay1);
        let area_b = (bx2 - bx1) * (by2 - by1);
        let union = area_a + area_b - inter;

        if union > 0.0 {
            inter / union
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub class_id: i32,
    pub class_name: String,
    pub bbox: BoundingBox,
    pub line_num: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub image: String,
    pub severity: IssueSeverity,
    #[serde(rename = "type")]
    pub issue_type: IssueType,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iou: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_num: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResult {
    pub filename: String,
    pub gt_count: usize,
    pub detection_count: usize,
    pub issues: Vec<Issue>,
}

impl ImageResult {
    pub fn new(filename: String, gt_count: usize, detection_count: usize) -> Self {
        Self {
            filename,
            gt_count,
            detection_count,
            issues: Vec::new(),
        }
    }

    pub fn add_issue(&mut self, issue: Issue) {
        self.issues.push(issue);
    }

    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }

    pub fn high_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::High)
            .count()
    }

    pub fn medium_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Medium)
            .count()
    }

    pub fn low_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Low)
            .count()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total_images: usize,
    pub images_audited: usize,
    pub images_with_issues: usize,
    pub total_issues: usize,
    pub by_severity: HashMap<String, usize>,
    pub by_type: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub generator: String,
    pub generator_version: String,
    pub generated_at: String,
    pub dataset_path: String,
    pub method: String,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub total_images: usize,
    pub images_audited: usize,
    #[serde(skip)]
    pub image_results: Vec<ImageResult>,
    pub summary: AuditSummary,
    pub flagged_images: Vec<ImageResult>,
}

/// Version from Cargo.toml
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

impl AuditResult {
    pub fn new(
        dataset_path: String,
        method: String,
        confidence_threshold: f32,
        iou_threshold: f32,
        total_images: usize,
        images_audited: usize,
    ) -> Self {
        Self {
            generator: "gt-audit".to_string(),
            generator_version: VERSION.to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            dataset_path,
            method,
            confidence_threshold,
            iou_threshold,
            total_images,
            images_audited,
            image_results: Vec::new(),
            summary: AuditSummary {
                total_images,
                images_audited,
                images_with_issues: 0,
                total_issues: 0,
                by_severity: HashMap::new(),
                by_type: HashMap::new(),
            },
            flagged_images: Vec::new(),
        }
    }

    pub fn add_image_result(&mut self, result: ImageResult) {
        if result.has_issues() {
            self.flagged_images.push(result.clone());
        }
        self.image_results.push(result);
        self.update_summary();
    }

    fn update_summary(&mut self) {
        self.summary.images_with_issues = self.image_results.iter().filter(|r| r.has_issues()).count();
        self.summary.total_issues = self.image_results.iter().map(|r| r.issues.len()).sum();

        // Count by severity
        let mut by_severity: HashMap<String, usize> = HashMap::new();
        let mut by_type: HashMap<String, usize> = HashMap::new();

        for result in &self.image_results {
            for issue in &result.issues {
                let sev = match issue.severity {
                    IssueSeverity::High => "high",
                    IssueSeverity::Medium => "medium",
                    IssueSeverity::Low => "low",
                };
                *by_severity.entry(sev.to_string()).or_insert(0) += 1;
                *by_type.entry(issue.issue_type.to_string()).or_insert(0) += 1;
            }
        }

        self.summary.by_severity = by_severity;
        self.summary.by_type = by_type;

        // Sort flagged images by issue count
        self.flagged_images.sort_by(|a, b| b.issues.len().cmp(&a.issues.len()));
    }

    pub fn images_with_issues(&self) -> usize {
        self.summary.images_with_issues
    }

    pub fn total_issues(&self) -> usize {
        self.summary.total_issues
    }

    pub fn high_count(&self) -> usize {
        *self.summary.by_severity.get("high").unwrap_or(&0)
    }

    pub fn medium_count(&self) -> usize {
        *self.summary.by_severity.get("medium").unwrap_or(&0)
    }

    pub fn low_count(&self) -> usize {
        *self.summary.by_severity.get("low").unwrap_or(&0)
    }

    pub fn issues_by_type(&self) -> Vec<(String, usize)> {
        let mut items: Vec<_> = self.summary.by_type.iter().map(|(k, v)| (k.clone(), *v)).collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items
    }
}
