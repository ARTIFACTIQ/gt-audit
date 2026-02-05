use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

mod dataset;
mod detector;
mod models;
mod report;

use dataset::YoloDataset;
use detector::{Detector, DetectorConfig, YoloDetector, ZeroShotDetector};
use models::AuditResult;
use report::{HtmlReporter, JsonReporter, Reporter};

#[derive(Parser)]
#[command(name = "gt-audit")]
#[command(author = "Artifactiq <hello@artifactiq.ai>")]
#[command(version = "0.1.0")]
#[command(about = "Fast ground truth label validation for object detection datasets")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate ground truth labels
    Validate {
        /// Path to dataset (YOLO format)
        #[arg(value_name = "DATASET")]
        dataset: PathBuf,

        /// Detection method: zero-shot, vlm, byom
        #[arg(short, long, default_value = "zero-shot")]
        method: String,

        /// Path to custom model (for byom method)
        #[arg(long)]
        model: Option<PathBuf>,

        /// Confidence threshold for detections
        #[arg(short, long, default_value = "0.25")]
        confidence: f32,

        /// IoU threshold for matching
        #[arg(long, default_value = "0.5")]
        iou: f32,

        /// Output file (json or html based on extension)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of images to sample (0 = all)
        #[arg(long, default_value = "0")]
        sample: usize,

        /// Random seed for sampling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Fail if high severity issues exceed threshold
        #[arg(long)]
        fail_on_high: Option<usize>,

        /// Fail if medium severity issues exceed threshold
        #[arg(long)]
        fail_on_medium: Option<usize>,

        /// Number of parallel workers
        #[arg(short = 'j', long)]
        workers: Option<usize>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Download required models
    Download {
        /// Model to download: clip, grounding-dino, all
        #[arg(default_value = "all")]
        model: String,
    },

    /// Show information about a dataset
    Info {
        /// Path to dataset
        #[arg(value_name = "DATASET")]
        dataset: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Validate {
            dataset,
            method,
            model,
            confidence,
            iou,
            output,
            sample,
            seed,
            fail_on_high,
            fail_on_medium,
            workers,
            verbose,
        } => {
            run_validate(
                dataset,
                method,
                model,
                confidence,
                iou,
                output,
                sample,
                seed,
                fail_on_high,
                fail_on_medium,
                workers,
                verbose,
            )
        }
        Commands::Download { model } => run_download(model),
        Commands::Info { dataset } => run_info(dataset),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_validate(
    dataset_path: PathBuf,
    method: String,
    model_path: Option<PathBuf>,
    confidence: f32,
    iou_threshold: f32,
    output: Option<PathBuf>,
    sample: usize,
    seed: u64,
    fail_on_high: Option<usize>,
    fail_on_medium: Option<usize>,
    workers: Option<usize>,
    _verbose: bool,
) -> Result<()> {
    let start = Instant::now();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║               gt-audit - Ground Truth Validator          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Load dataset
    println!("📂 Loading dataset: {}", dataset_path.display());
    let dataset = YoloDataset::load(&dataset_path)?;
    println!("   Classes: {}", dataset.class_names.len());
    println!("   Images: {}", dataset.image_count());

    // Get images to process
    let mut images = dataset.get_images();
    if sample > 0 && sample < images.len() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        images.shuffle(&mut rng);
        images.truncate(sample);
        println!("   Sampled: {} images (seed={})", sample, seed);
    }

    // Initialize detector
    println!();

    // Get class names sorted by ID for the model
    let mut model_class_names: Vec<String> = Vec::new();
    let mut class_ids: Vec<i32> = dataset.class_names.keys().copied().collect();
    class_ids.sort();
    for id in class_ids {
        model_class_names.push(dataset.class_names.get(&id).cloned().unwrap_or_default());
    }

    // Determine which method to use
    let effective_method = if model_path.is_some() {
        "yolo".to_string()
    } else {
        method.clone()
    };

    println!("🔍 Initializing detector: {}", effective_method);
    let config = DetectorConfig {
        confidence_threshold: confidence,
        iou_threshold,
        model_path: model_path.clone(),
    };

    let detector: Box<dyn Detector + Sync> = match effective_method.as_str() {
        "yolo" | "byom" => {
            if config.model_path.is_none() {
                anyhow::bail!("YOLO/BYOM method requires --model path to ONNX model");
            }
            Box::new(YoloDetector::new(config, model_class_names)?)
        }
        "zero-shot" => Box::new(ZeroShotDetector::new(config)?),
        "vlm" => {
            println!("   VLM method requires GPU, checking...");
            anyhow::bail!("VLM method not yet implemented - coming soon");
        }
        _ => anyhow::bail!("Unknown method: {}. Use: zero-shot, vlm, yolo", method),
    };

    // Set up parallelism
    if let Some(w) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(w)
            .build_global()
            .ok();
    }

    // Progress bar
    let pb = ProgressBar::new(images.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    // Process images in parallel
    println!();
    println!("🔬 Auditing {} images...", images.len());

    let results: Vec<_> = images
        .par_iter()
        .map(|img_path| {
            let annotations = dataset.load_annotations(img_path);
            let result = detector.audit_image(img_path, &annotations, &dataset.class_names);
            pb.inc(1);
            result
        })
        .collect();

    pb.finish_with_message("Done!");

    // Build audit result
    let mut audit_result = AuditResult::new(
        dataset_path.to_string_lossy().to_string(),
        effective_method.clone(),
        confidence,
        iou_threshold,
        dataset.image_count(),
        images.len(),
    );

    for result in results.into_iter().flatten() {
        audit_result.add_image_result(result);
    }

    // Print summary
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                      AUDIT SUMMARY                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Total images:       {}", audit_result.total_images);
    println!("  Images audited:     {}", audit_result.images_audited);
    println!("  Images with issues: {}", audit_result.images_with_issues());
    println!("  Total issues:       {}", audit_result.total_issues());
    println!();
    println!("  By severity:");
    println!("    🔴 High:   {}", audit_result.high_count());
    println!("    🟡 Medium: {}", audit_result.medium_count());
    println!("    ⚪ Low:    {}", audit_result.low_count());
    println!();
    println!("  By type:");
    for (issue_type, count) in audit_result.issues_by_type() {
        println!("    {}: {}", issue_type, count);
    }
    println!();
    println!("  Time: {:.2}s", start.elapsed().as_secs_f64());

    // Save output
    if let Some(output_path) = &output {
        let ext = output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json");

        match ext {
            "html" => {
                let reporter = HtmlReporter::new();
                reporter.generate(&audit_result, output_path)?;
                println!("📄 HTML report saved: {}", output_path.display());
            }
            _ => {
                let reporter = JsonReporter::new();
                reporter.generate(&audit_result, output_path)?;
                println!("📄 JSON report saved: {}", output_path.display());
            }
        }
    }

    // Check thresholds for CI
    let mut exit_code = 0;

    if let Some(threshold) = fail_on_high {
        if audit_result.high_count() > threshold {
            eprintln!(
                "❌ FAIL: High severity issues ({}) exceed threshold ({})",
                audit_result.high_count(),
                threshold
            );
            exit_code = 1;
        }
    }

    if let Some(threshold) = fail_on_medium {
        let medium_plus = audit_result.high_count() + audit_result.medium_count();
        if medium_plus > threshold {
            eprintln!(
                "❌ FAIL: Medium+ severity issues ({}) exceed threshold ({})",
                medium_plus, threshold
            );
            exit_code = 1;
        }
    }

    if exit_code == 0 && (fail_on_high.is_some() || fail_on_medium.is_some()) {
        println!("✅ PASS: Issue counts within thresholds");
    }

    std::process::exit(exit_code);
}

fn run_download(model: String) -> Result<()> {
    println!("📥 Downloading models...");

    match model.as_str() {
        "clip" | "all" => {
            println!("   Downloading CLIP model...");
            detector::download_clip_model()?;
            println!("   ✓ CLIP model ready");
        }
        "grounding-dino" => {
            println!("   Downloading Grounding DINO model...");
            println!("   ⚠ Grounding DINO - coming soon");
        }
        _ => {
            anyhow::bail!("Unknown model: {}. Use: clip, grounding-dino, all", model);
        }
    }

    println!("✅ Done!");
    Ok(())
}

fn run_info(dataset_path: PathBuf) -> Result<()> {
    let dataset = YoloDataset::load(&dataset_path)?;

    println!("Dataset: {}", dataset_path.display());
    println!("Images: {}", dataset.image_count());
    println!("Classes: {}", dataset.class_names.len());
    println!();
    println!("Class names:");
    for (id, name) in &dataset.class_names {
        println!("  {}: {}", id, name);
    }

    Ok(())
}
