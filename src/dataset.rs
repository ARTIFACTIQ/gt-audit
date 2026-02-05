//! YOLO dataset loading

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::models::{Annotation, BoundingBox};

pub struct YoloDataset {
    pub path: PathBuf,
    pub class_names: HashMap<i32, String>,
    pub images_dir: PathBuf,
    pub labels_dir: PathBuf,
}

impl YoloDataset {
    pub fn load(path: &Path) -> Result<Self> {
        let path = path.to_path_buf();

        // Detect structure and load class names
        let (images_dir, labels_dir) = Self::detect_structure(&path)?;
        let class_names = Self::load_class_names(&path)?;

        Ok(Self {
            path,
            class_names,
            images_dir,
            labels_dir,
        })
    }

    fn detect_structure(path: &Path) -> Result<(PathBuf, PathBuf)> {
        // Try standard YOLO structure: images/val, labels/val
        for split in ["val", "train", "test", ""] {
            let img_dir = if split.is_empty() {
                path.join("images")
            } else {
                path.join("images").join(split)
            };

            let lbl_dir = if split.is_empty() {
                path.join("labels")
            } else {
                path.join("labels").join(split)
            };

            // Also check data/val/labels structure
            let lbl_dir_alt = if split.is_empty() {
                path.join("data").join("labels")
            } else {
                path.join("data").join(split).join("labels")
            };

            if img_dir.exists() {
                if lbl_dir.exists() {
                    return Ok((img_dir, lbl_dir));
                } else if lbl_dir_alt.exists() {
                    return Ok((img_dir, lbl_dir_alt));
                }
            }
        }

        // Try flat structure
        if path.join("images").exists() && path.join("labels").exists() {
            return Ok((path.join("images"), path.join("labels")));
        }

        anyhow::bail!(
            "Could not detect dataset structure in {}. Expected images/ and labels/ directories.",
            path.display()
        )
    }

    fn load_class_names(path: &Path) -> Result<HashMap<i32, String>> {
        let mut class_names = HashMap::new();

        // Try dataset.yaml in multiple locations
        let yaml_paths = [
            path.join("dataset.yaml"),
            path.join("data").join("dataset.yaml"),
            path.join("data.yaml"),
        ];

        for yaml_path in yaml_paths {
            if !yaml_path.exists() {
                continue;
            }
            let content = fs::read_to_string(&yaml_path)
                .with_context(|| format!("Failed to read {}", yaml_path.display()))?;

            let yaml: serde_yaml::Value = serde_yaml::from_str(&content)
                .with_context(|| format!("Failed to parse {}", yaml_path.display()))?;

            if let Some(names) = yaml.get("names") {
                if let Some(seq) = names.as_sequence() {
                    for (i, name) in seq.iter().enumerate() {
                        if let Some(s) = name.as_str() {
                            class_names.insert(i as i32, s.to_string());
                        }
                    }
                } else if let Some(map) = names.as_mapping() {
                    for (k, v) in map {
                        if let (Some(id), Some(name)) = (k.as_i64(), v.as_str()) {
                            class_names.insert(id as i32, name.to_string());
                        }
                    }
                }
            }

            if !class_names.is_empty() {
                return Ok(class_names);
            }
        }


        // Try classes.txt
        let txt_path = path.join("classes.txt");
        if txt_path.exists() {
            let content = fs::read_to_string(&txt_path)?;
            for (i, line) in content.lines().enumerate() {
                let name = line.trim();
                if !name.is_empty() {
                    class_names.insert(i as i32, name.to_string());
                }
            }

            if !class_names.is_empty() {
                return Ok(class_names);
            }
        }

        eprintln!("Warning: No class names found, using class IDs");
        Ok(class_names)
    }

    pub fn image_count(&self) -> usize {
        self.get_images().len()
    }

    pub fn get_images(&self) -> Vec<PathBuf> {
        let mut images = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.images_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ["jpg", "jpeg", "png", "webp", "bmp"].contains(&ext.as_str()) {
                        images.push(path);
                    }
                }
            }
        }

        images.sort();
        images
    }

    pub fn get_label_path(&self, image_path: &Path) -> PathBuf {
        let stem = image_path.file_stem().unwrap_or_default();
        self.labels_dir.join(format!("{}.txt", stem.to_string_lossy()))
    }

    pub fn load_annotations(&self, image_path: &Path) -> Vec<Annotation> {
        let label_path = self.get_label_path(image_path);

        if !label_path.exists() {
            return Vec::new();
        }

        let content = match fs::read_to_string(&label_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut annotations = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() >= 5 {
                if let (Ok(class_id), Ok(x), Ok(y), Ok(w), Ok(h)) = (
                    parts[0].parse::<i32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                    parts[3].parse::<f32>(),
                    parts[4].parse::<f32>(),
                ) {
                    let class_name = self
                        .class_names
                        .get(&class_id)
                        .cloned()
                        .unwrap_or_else(|| format!("class_{}", class_id));

                    annotations.push(Annotation {
                        class_id,
                        class_name,
                        bbox: BoundingBox::new(x, y, w, h),
                        line_num: line_num + 1,
                    });
                }
            }
        }

        annotations
    }

    pub fn get_class_name(&self, class_id: i32) -> String {
        self.class_names
            .get(&class_id)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", class_id))
    }
}
