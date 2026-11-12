//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Trace a PNG into a vector bitmap and write it out as SVG.
//!
//! The example shows the full path an application takes when it wants to turn
//! a raster asset into scalable artwork:
//!
//! 1. Decode the PNG with the `image` crate into an RGBA buffer.
//! 2. Wrap that buffer in a [`ColorImage`], the bitmap model the tracer
//!    consumes.
//! 3. Trace it into a [`VectorDoc`] of filled shapes whose paths are made of
//!    absolute line and cubic Bézier segments.
//! 4. Serialize the document to SVG XML and write it to a file.
//!
//! `codevar-bmpvector` deliberately does not decode images and does not emit
//! SVG, so both the decode step and the serializer live here.
//!
//! ```text
//! cargo run --example png_to_svg -p codevar-bmpvector
//! cargo run --example png_to_svg -p codevar-bmpvector -- <input.png> <output.svg>
//! ```
//!
//! Without arguments the example traces the bundled
//! `examples/resources/gopher.png` and writes `gopher.svg` next to it.

use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use codevar_bmpvector::{Color, ColorImage, Config, FillRule, Preset, Segment, SubPath, VectorDoc};

/// The image shipped with the example, used when no input path is given.
const DEFAULT_INPUT: &str = "examples/resources/gopher.png";

/// The SVG file written when no output path is given.
const DEFAULT_OUTPUT: &str = "gopher.svg";

fn main() -> ExitCode {
    let (input, output) = match parse_arguments() {
        Ok(paths) => paths,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("usage: png_to_svg [input.png] [output.svg]");
            return ExitCode::FAILURE;
        }
    };

    match run(&input, &output) {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("png_to_svg failed: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Resolves the input and output paths from the command line.
fn parse_arguments() -> Result<(PathBuf, PathBuf), String> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .map_or_else(|| PathBuf::from(DEFAULT_INPUT), PathBuf::from);
    let output = arguments
        .next()
        .map_or_else(|| PathBuf::from(DEFAULT_OUTPUT), PathBuf::from);

    if arguments.next().is_some() {
        return Err("too many arguments".to_owned());
    }

    Ok((input, output))
}

/// Runs the full decode, trace and serialize pipeline.
fn run(input: &Path, output: &Path) -> Result<String, Box<dyn Error>> {
    let bitmap = load_bitmap(input)?;
    let config = tracing_config();
    let (document, stats) = config.trace_with_stats(&bitmap)?;

    let svg = to_svg(&document);
    fs::write(output, svg.as_bytes())
        .map_err(|error| format!("could not write {}: {error}", output.display()))?;

    Ok(format!(
        "{} ({}x{}) -> {}: {} shapes, {} subpaths, {} segments ({} lines, {} cubics), \
         {} palette entries",
        input.display(),
        document.width,
        document.height,
        output.display(),
        stats.document.shapes,
        stats.document.subpaths,
        stats.document.segments,
        stats.document.line_segments,
        stats.document.cubic_segments,
        stats.palette_entries,
    ))
}

/// Decodes a PNG file into the bitmap model used by the tracer.
///
/// The `image` crate produces a tightly packed, row-major RGBA buffer, which
/// is exactly the layout [`ColorImage::from_rgba8`] expects.
fn load_bitmap(path: &Path) -> Result<ColorImage, Box<dyn Error>> {
    let decoded = image::open(path)?;
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ColorImage::from_rgba8(width, height, rgba.as_raw())?)
}

/// Picks the tracing configuration for the example.
///
/// The bundled artwork is a flat logo, so the logo preset fits it well: few
/// palette entries, crisp corners and aggressive despeckling. Transparency is
/// flattened onto white so the output is a plain opaque illustration.
fn tracing_config() -> Config {
    Config::preset(Preset::Logo).with_background(Color::WHITE)
}

/// Serializes a traced document as SVG XML.
///
/// `codevar-bmpvector` produces a data model rather than a file format, so the
/// conversion to SVG happens here. The coordinate convention matches SVG's:
/// `x` grows to the right, `y` grows down, and the origin is the top-left
/// corner, so no axis flip is needed.
fn to_svg(document: &VectorDoc) -> String {
    let mut svg = String::with_capacity(4096);

    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        document.width, document.height, document.width, document.height
    );

    // Shapes are stored in paint order: the first shape is drawn first and
    // later shapes may overlap it.
    for shape in &document.shapes {
        let data = path_data(&shape.path.subpaths);
        if data.is_empty() {
            continue;
        }

        let _ = writeln!(
            svg,
            "  <path fill=\"{}\" fill-opacity=\"{}\" fill-rule=\"{}\" d=\"{}\"/>",
            rgb_hex(shape.fill.color),
            alpha_opacity(shape.fill.color),
            fill_rule_name(shape.fill.rule),
            data
        );
    }

    svg.push_str("</svg>\n");
    svg
}

/// Builds the `d` attribute for a compound path.
///
/// Every subpath becomes one `M` followed by its segments: straight lines
/// become `L` and cubic Béziers become `C`. Closed subpaths end with `Z` so
/// the ring is filled rather than implicitly joined. Because all the rings of
/// a shape share one path, the fill rule decides which of them are holes.
fn path_data(subpaths: &[SubPath]) -> String {
    let mut data = String::new();

    for subpath in subpaths {
        if subpath.is_empty() {
            continue;
        }

        let start = subpath.start;
        let _ = write!(data, "M {} {}", start.x, start.y);

        for segment in &subpath.segments {
            match segment {
                Segment::Line { end, .. } => {
                    let _ = write!(data, " L {} {}", end.x, end.y);
                }
                Segment::Cubic(cubic) => {
                    let _ = write!(
                        data,
                        " C {} {} {} {} {} {}",
                        cubic.control1.x,
                        cubic.control1.y,
                        cubic.control2.x,
                        cubic.control2.y,
                        cubic.end.x,
                        cubic.end.y
                    );
                }
            }
        }

        if subpath.closed {
            data.push_str(" Z");
        }
    }

    data
}

/// Formats the opaque part of a color as `#rrggbb`.
fn rgb_hex(color: Color) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
}

/// Formats the alpha channel as an SVG opacity value.
///
/// SVG 1.1 has no eight-digit hex color, so the alpha channel is written as a
/// separate `fill-opacity` attribute instead.
fn alpha_opacity(color: Color) -> String {
    format!("{:.3}", f64::from(color.a) / 255.0)
}

/// Names the fill rule used by SVG.
fn fill_rule_name(rule: FillRule) -> &'static str {
    match rule {
        FillRule::EvenOdd => "evenodd",
        FillRule::NonZero => "nonzero",
    }
}
