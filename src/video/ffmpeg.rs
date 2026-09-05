//! FFmpeg wrapper for video compilation.

use crate::config::VideoConfig;
use crate::error::{Error, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Compile images into a timelapse video.
///
/// # Arguments
/// * `input_dir` - Directory containing JPEG images (sorted alphabetically for order)
/// * `output_path` - Path for the output video file
/// * `config` - Video configuration
/// * `cancel_token` - Optional cancellation token to kill the ffmpeg process
/// * `progress_callback` - Called with (current_frame, total_frames)
pub async fn compile_timelapse<F>(
    input_dir: &Path,
    output_path: &Path,
    config: &VideoConfig,
    cancel_token: Option<&CancellationToken>,
    mut progress_callback: F,
) -> Result<()>
where
    F: FnMut(u32, u32),
{
    // Collect and sort image files for reliable ordering.
    // Filenames are timestamp-based, so alphabetical sort = chronological order.
    let mut image_files: Vec<_> = Vec::new();
    let mut entries = tokio::fs::read_dir(input_dir).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().map(|e| e == "jpg").unwrap_or(false) {
            image_files.push(path);
        }
    }

    if image_files.is_empty() {
        return Err(Error::VideoCompilation(
            "No images found in input directory".to_string(),
        ));
    }

    // Sort alphabetically (timestamp-based names ensure chronological order)
    image_files.sort();
    let image_count = image_files.len() as u32;

    tracing::info!("Compiling {} images into video", image_count);

    // Create concat file for ffmpeg (more reliable than glob pattern).
    // Format: file '/path/to/image.jpg'\nduration 0.0667\n...
    let frame_duration = 1.0 / config.framerate as f64;
    let concat_path = input_dir.join(".concat.txt");

    let mut concat_content = String::new();
    for (i, path) in image_files.iter().enumerate() {
        // Use just the filename since concat file is in the same directory as images
        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy())
            .unwrap_or_default()
            .replace('\'', "'\\''");
        concat_content.push_str(&format!("file '{}'\n", filename));

        // Don't add duration after the last frame
        if i < image_files.len() - 1 {
            concat_content.push_str(&format!("duration {:.6}\n", frame_duration));
        }
    }

    // Write concat file
    let mut file = tokio::fs::File::create(&concat_path).await?;
    file.write_all(concat_content.as_bytes()).await?;
    file.flush().await?;

    // Use a closure to ensure concat file is cleaned up on all paths
    let result = run_ffmpeg(
        &concat_path,
        &image_files,
        output_path,
        config,
        cancel_token,
        image_count,
        &mut progress_callback,
    )
    .await;

    // Always clean up concat file, even on error
    let _ = tokio::fs::remove_file(&concat_path).await;

    result
}

/// Run the ffmpeg process and handle progress/cancellation.
async fn run_ffmpeg<F>(
    concat_path: &Path,
    image_files: &[std::path::PathBuf],
    output_path: &Path,
    config: &VideoConfig,
    cancel_token: Option<&CancellationToken>,
    image_count: u32,
    progress_callback: &mut F,
) -> Result<()>
where
    F: FnMut(u32, u32),
{
    let frame_duration = 1.0 / config.framerate as f64;
    let dissolve_duration = config.dissolve_duration as f64;
    let stream_duration = frame_duration + dissolve_duration;
    let transition_framerate = config.transition_framerate;
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");

    if config.dissolve && image_files.len() > 1 {
        // Give each still enough time for the configured crossfade. Streams
        // advance by one frame interval, so transitions overlap naturally.
        let mut filter = String::new();
        for (index, _) in image_files.iter().enumerate() {
            // Alternate the sign to create a subtle warp between frames rather
            // than applying the exact same distortion to every image.
            let warp_sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            let warp = config.warp as f64 * warp_sign;
            filter.push_str(&format!(
                "[{}:v]lenscorrection=k1={:.4}:k2={:.4},trim=duration={:.6},setpts=PTS-STARTPTS[v{}];",
                index,
                warp * 0.1,
                warp * 0.02,
                stream_duration,
                index
            ));
        }
        let mut current = "v0".to_string();
        for index in 1..image_files.len() {
            let output = format!("x{}", index);
            filter.push_str(&format!(
                "[{}][v{}]xfade=transition=fade:duration={:.6}:offset={:.6}[{}];",
                current,
                index,
                dissolve_duration,
                frame_duration * index as f64,
                output
            ));
            current = output;
        }

        for path in image_files {
            cmd.arg("-loop")
                .arg("1")
                .arg("-framerate")
                .arg(transition_framerate.to_string())
                .arg("-i")
                .arg(path);
        }
        cmd.arg("-filter_complex")
            .arg(filter)
            .arg("-map")
            .arg(format!("[{}]", current));
    } else {
        cmd.arg("-f")
            .arg("concat")
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(concat_path);
    }

    cmd.arg("-r")
        .arg(if config.dissolve && image_files.len() > 1 {
            transition_framerate.to_string()
        } else {
            config.framerate.to_string()
        })
        .arg("-c:v")
        .arg(&config.codec)
        .arg("-crf")
        .arg(config.crf.to_string())
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-progress")
        .arg("pipe:1")
        .arg(output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| Error::FFmpeg(format!("Failed to spawn ffmpeg (is it installed?): {}", e)))?;

    // Capture stdout for progress and stderr for errors
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::FFmpeg("Failed to capture ffmpeg stdout".to_string()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| Error::FFmpeg("Failed to capture ffmpeg stderr".to_string()))?;

    // Read stderr in background to avoid blocking
    let stderr_handle = tokio::spawn(async move {
        let mut stderr_reader = BufReader::new(stderr).lines();
        let mut stderr_output = String::new();
        while let Ok(Some(line)) = stderr_reader.next_line().await {
            stderr_output.push_str(&line);
            stderr_output.push('\n');
        }
        stderr_output
    });

    // Parse progress from stdout, with cancellation support
    let mut reader = BufReader::new(stdout).lines();

    // Create a dummy token for when no cancel_token is provided
    let owned_token = CancellationToken::new();
    let token = cancel_token.unwrap_or(&owned_token);

    loop {
        tokio::select! {
            _ = token.cancelled() => {
                // Kill the ffmpeg process on cancellation
                let _ = child.kill().await;
                return Err(Error::Cancelled);
            }
            line = reader.next_line() => {
                match line? {
                    Some(line) => {
                        if line.starts_with("frame=") {
                            if let Some(frame_str) = line.strip_prefix("frame=") {
                                if let Ok(frame) = frame_str.trim().parse::<u32>() {
                                    let clamped_frame = frame.min(image_count);
                                    progress_callback(clamped_frame, image_count);
                                }
                            }
                        }
                    }
                    None => break, // EOF
                }
            }
        }
    }

    let status = child.wait().await?;
    let stderr_output = stderr_handle.await.unwrap_or_default();

    if !status.success() {
        // Get last few lines of stderr for the error message
        let error_lines: Vec<&str> = stderr_output.lines().rev().take(10).collect();
        let error_detail = error_lines.into_iter().rev().collect::<Vec<_>>().join("\n");

        return Err(Error::FFmpeg(format!(
            "ffmpeg exited with status {}\n{}",
            status, error_detail
        )));
    }

    progress_callback(image_count, image_count);
    tracing::info!("Video compilation complete: {:?}", output_path);

    Ok(())
}

/// Check if ffmpeg is available.
pub async fn check_ffmpeg() -> Result<String> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .await
        .map_err(|e| Error::FFmpeg(format!("ffmpeg not found: {}", e)))?;

    if !output.status.success() {
        return Err(Error::FFmpeg("ffmpeg version check failed".to_string()));
    }

    let version_output = String::from_utf8_lossy(&output.stdout);
    let version = version_output
        .lines()
        .next()
        .unwrap_or("unknown")
        .to_string();

    Ok(version)
}
