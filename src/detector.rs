//! Detection methods for ground truth validation

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ndarray::{Array4, ArrayD};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::models::{
    Annotation, BoundingBox, Detection, ImageResult, Issue, IssueSeverity, IssueType,
};

/// Configuration for detectors
pub struct DetectorConfig {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub model_path: Option<PathBuf>,
}

/// Trait for detection methods
pub trait Detector: Send {
    fn audit_image(
        &self,
        image_path: &Path,
        annotations: &[Annotation],
        class_names: &HashMap<i32, String>,
    ) -> Result<ImageResult>;

    fn detect(&self, image: &DynamicImage, class_names: &[String]) -> Result<Vec<Detection>>;
}

/// YOLO-based detector using ONNX Runtime
pub struct YoloDetector {
    session: Mutex<Session>,
    config: DetectorConfig,
    model_class_names: Vec<String>,
}

impl YoloDetector {
    pub fn new(config: DetectorConfig, model_class_names: Vec<String>) -> Result<Self> {
        let model_path = config
            .model_path
            .as_ref()
            .context("Model path required for YOLO detector")?;

        println!("   Loading ONNX model: {}", model_path.display());

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        println!("   Model loaded successfully");

        Ok(Self {
            session: Mutex::new(session),
            config,
            model_class_names,
        })
    }

    fn preprocess_image(&self, image: &DynamicImage) -> Array4<f32> {
        let img_size = 640u32;
        let resized = image.resize_exact(img_size, img_size, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let mut input = Array4::<f32>::zeros((1, 3, img_size as usize, img_size as usize));

        for (x, y, pixel) in rgb.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }

        input
    }

    fn postprocess_detections(
        &self,
        output: &ArrayD<f32>,
        _orig_width: u32,
        _orig_height: u32,
    ) -> Vec<Detection> {
        let mut detections = Vec::new();
        let conf_threshold = self.config.confidence_threshold;

        // YOLOv8 output format: [1, num_classes + 4, num_detections]
        // Transpose to [num_detections, num_classes + 4]
        let output = output.view();

        if output.ndim() < 2 {
            return detections;
        }

        let shape = output.shape();
        let num_classes = if shape.len() == 3 {
            shape[1] - 4 // [1, classes+4, detections]
        } else {
            return detections;
        };

        // Process each detection
        let num_detections = shape[2];
        for i in 0..num_detections {
            // Get box coordinates (x_center, y_center, width, height)
            let x = output[[0, 0, i]];
            let y = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];

            // Find class with highest confidence
            // Note: YOLOv8/v11 ONNX exports already apply sigmoid internally
            let mut max_conf = 0.0f32;
            let mut max_class = 0usize;
            for c in 0..num_classes {
                let conf = output[[0, 4 + c, i]];
                if conf > max_conf {
                    max_conf = conf;
                    max_class = c;
                }
            }

            if max_conf >= conf_threshold {
                // Normalize coordinates to [0, 1]
                let norm_x = x / 640.0;
                let norm_y = y / 640.0;
                let norm_w = w / 640.0;
                let norm_h = h / 640.0;

                let class_name = self
                    .model_class_names
                    .get(max_class)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", max_class));

                detections.push(Detection {
                    class_name,
                    confidence: max_conf,
                    bbox: BoundingBox::new(norm_x, norm_y, norm_w, norm_h),
                });
            }
        }

        // Apply NMS
        detections = self.non_max_suppression(detections);
        detections
    }

    fn non_max_suppression(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }
            keep.push(detections[i].clone());

            for j in (i + 1)..detections.len() {
                if suppressed[j] {
                    continue;
                }
                if detections[i].class_name == detections[j].class_name {
                    let iou = detections[i].bbox.iou(&detections[j].bbox);
                    if iou > self.config.iou_threshold {
                        suppressed[j] = true;
                    }
                }
            }
        }

        keep
    }

    fn classes_equivalent(class1: &str, class2: &str) -> bool {
        let c1 = class1.to_lowercase();
        let c2 = class2.to_lowercase();

        if c1 == c2 {
            return true;
        }

        let groups: &[&[&str]] = &[
            &[
                "clothing", "dress", "suit", "jacket", "coat", "top", "shirt", "blouse",
            ],
            &["footwear", "boot", "shoe", "sandal", "high heels", "sneaker"],
            &[
                "bag",
                "handbag",
                "backpack",
                "briefcase",
                "luggage and bags",
                "purse",
            ],
            &["pants", "jeans", "trousers", "shorts"],
            &["person", "human", "man", "woman", "people", "boy", "girl"],
        ];

        for group in groups {
            let in_group1 = group.iter().any(|&g| c1.contains(g) || g.contains(&c1));
            let in_group2 = group.iter().any(|&g| c2.contains(g) || g.contains(&c2));
            if in_group1 && in_group2 {
                return true;
            }
        }

        false
    }
}

impl Detector for YoloDetector {
    fn audit_image(
        &self,
        image_path: &Path,
        annotations: &[Annotation],
        _class_names: &HashMap<i32, String>,
    ) -> Result<ImageResult> {
        let filename = image_path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        // Load image
        let image = image::open(image_path).context("Failed to load image")?;
        let (_orig_width, _orig_height) = image.dimensions();

        // Run detection
        let detections = self.detect(&image, &[])?;

        let mut result = ImageResult::new(filename.clone(), annotations.len(), detections.len());

        // Track matched GT annotations
        let mut matched_gt: Vec<bool> = vec![false; annotations.len()];

        // Check each detection against GT
        for det in &detections {
            let mut best_iou = 0.0f32;
            let mut best_gt_idx: Option<usize> = None;

            for (idx, ann) in annotations.iter().enumerate() {
                let iou = det.bbox.iou(&ann.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_gt_idx = Some(idx);
                }
            }

            if best_iou >= 0.3 {
                if let Some(gt_idx) = best_gt_idx {
                    let gt = &annotations[gt_idx];
                    matched_gt[gt_idx] = true;

                    // Check for class mismatch
                    if !Self::classes_equivalent(&det.class_name, &gt.class_name) {
                        result.add_issue(Issue {
                            image: filename.clone(),
                            severity: IssueSeverity::High,
                            issue_type: IssueType::ClassMismatch,
                            description: format!(
                                "Model detects '{}' ({:.1}%), GT says '{}'",
                                det.class_name,
                                det.confidence * 100.0,
                                gt.class_name
                            ),
                            gt_class: Some(gt.class_name.clone()),
                            detected_class: Some(det.class_name.clone()),
                            confidence: Some(det.confidence),
                            iou: Some(best_iou),
                            explanation: None,
                            line_num: Some(gt.line_num),
                        });
                    }
                }
            } else {
                // Detection with no matching GT - possible missing label
                result.add_issue(Issue {
                    image: filename.clone(),
                    severity: IssueSeverity::Medium,
                    issue_type: IssueType::MissingLabel,
                    description: format!(
                        "Model detects '{}' ({:.1}%) with no matching GT",
                        det.class_name,
                        det.confidence * 100.0
                    ),
                    gt_class: None,
                    detected_class: Some(det.class_name.clone()),
                    confidence: Some(det.confidence),
                    iou: None,
                    explanation: None,
                    line_num: None,
                });
            }
        }

        // Check for phantom GT (GT with no detection)
        for (idx, ann) in annotations.iter().enumerate() {
            if !matched_gt[idx] {
                result.add_issue(Issue {
                    image: filename.clone(),
                    severity: IssueSeverity::Low,
                    issue_type: IssueType::SpuriousLabel,
                    description: format!("GT has '{}' but model detects nothing there", ann.class_name),
                    gt_class: Some(ann.class_name.clone()),
                    detected_class: None,
                    confidence: None,
                    iou: None,
                    explanation: None,
                    line_num: Some(ann.line_num),
                });
            }
        }

        Ok(result)
    }

    fn detect(&self, image: &DynamicImage, _class_names: &[String]) -> Result<Vec<Detection>> {
        let (orig_width, orig_height) = image.dimensions();

        // Preprocess
        let input = self.preprocess_image(image);

        // Create shape and flattened data for ort
        let shape: Vec<i64> = input.shape().iter().map(|&x| x as i64).collect();
        let data: Vec<f32> = input.into_raw_vec_and_offset().0;

        // Create input tensor from shape and data
        let input_tensor = ort::value::Tensor::from_array((shape.clone(), data))?;

        // Run inference (lock the session for thread safety)
        let mut session = self.session.lock().map_err(|e| anyhow::anyhow!("Failed to lock session: {}", e))?;
        let outputs = session.run(ort::inputs!["images" => input_tensor])?;

        // Get output tensor
        let binding = outputs["output0"].try_extract_tensor::<f32>()?;
        let (out_shape, out_data) = binding;

        // Convert to ndarray for processing
        let output = ArrayD::from_shape_vec(
            out_shape.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            out_data.to_vec()
        )?;

        // Postprocess
        let detections = self.postprocess_detections(&output, orig_width, orig_height);

        Ok(detections)
    }
}

/// Zero-shot detector using basic heuristics (fallback when no model provided)
pub struct ZeroShotDetector {
    config: DetectorConfig,
}

impl ZeroShotDetector {
    pub fn new(config: DetectorConfig) -> Result<Self> {
        println!("   Using heuristic-based validation (no model provided)");
        println!("   For better results, use --model with a trained YOLO model");
        Ok(Self { config })
    }
}

impl Detector for ZeroShotDetector {
    fn audit_image(
        &self,
        image_path: &Path,
        annotations: &[Annotation],
        _class_names: &HashMap<i32, String>,
    ) -> Result<ImageResult> {
        let filename = image_path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut result = ImageResult::new(filename.clone(), annotations.len(), 0);

        // Load image
        let image = match image::open(image_path) {
            Ok(img) => img,
            Err(e) => {
                result.add_issue(Issue {
                    image: filename,
                    severity: IssueSeverity::High,
                    issue_type: IssueType::ClassMismatch,
                    description: format!("Failed to load image: {}", e),
                    gt_class: None,
                    detected_class: None,
                    confidence: None,
                    iou: None,
                    explanation: None,
                    line_num: None,
                });
                return Ok(result);
            }
        };

        // Validate each annotation
        for ann in annotations {
            // Check bbox validity
            let bbox_valid = ann.bbox.w > 0.0
                && ann.bbox.h > 0.0
                && ann.bbox.w <= 1.0
                && ann.bbox.h <= 1.0
                && ann.bbox.x >= 0.0
                && ann.bbox.y >= 0.0
                && ann.bbox.x <= 1.0
                && ann.bbox.y <= 1.0
                && (ann.bbox.x - ann.bbox.w / 2.0) >= -0.01
                && (ann.bbox.y - ann.bbox.h / 2.0) >= -0.01;

            if !bbox_valid {
                result.add_issue(Issue {
                    image: filename.clone(),
                    severity: IssueSeverity::High,
                    issue_type: IssueType::Localization,
                    description: format!(
                        "Invalid bbox for '{}': x={:.3}, y={:.3}, w={:.3}, h={:.3}",
                        ann.class_name, ann.bbox.x, ann.bbox.y, ann.bbox.w, ann.bbox.h
                    ),
                    gt_class: Some(ann.class_name.clone()),
                    detected_class: None,
                    confidence: None,
                    iou: None,
                    explanation: Some("Bounding box coordinates out of valid range".to_string()),
                    line_num: Some(ann.line_num),
                });
            }
        }

        // Check for no annotations
        if annotations.is_empty() {
            let (width, height) = image.dimensions();
            if width > 100 && height > 100 {
                result.add_issue(Issue {
                    image: filename.clone(),
                    severity: IssueSeverity::Low,
                    issue_type: IssueType::SpuriousLabel,
                    description: "Image has no annotations".to_string(),
                    gt_class: None,
                    detected_class: None,
                    confidence: None,
                    iou: None,
                    explanation: None,
                    line_num: None,
                });
            }
        }

        Ok(result)
    }

    fn detect(&self, _image: &DynamicImage, _class_names: &[String]) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
}

/// Download CLIP model for zero-shot detection
pub fn download_clip_model() -> Result<()> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gt-audit")
        .join("models");

    std::fs::create_dir_all(&cache_dir)?;

    println!("   Model cache directory: {}", cache_dir.display());
    println!("   For now, use --model with your trained YOLO model");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_equivalence() {
        assert!(YoloDetector::classes_equivalent("Dress", "Clothing"));
        assert!(YoloDetector::classes_equivalent("Boot", "Footwear"));
        assert!(!YoloDetector::classes_equivalent("Boot", "Dress"));
        assert!(YoloDetector::classes_equivalent("person", "Person"));
        assert!(YoloDetector::classes_equivalent("Man", "Person"));
    }

    #[test]
    fn test_bbox_iou() {
        let box1 = BoundingBox::new(0.5, 0.5, 0.4, 0.4);
        let box2 = BoundingBox::new(0.5, 0.5, 0.4, 0.4);
        assert!((box1.iou(&box2) - 1.0).abs() < 0.001);

        let box3 = BoundingBox::new(0.0, 0.0, 0.2, 0.2);
        assert!(box1.iou(&box3) < 0.1);
    }
}
