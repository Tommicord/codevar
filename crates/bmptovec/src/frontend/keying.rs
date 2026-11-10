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

//! Image keying: detect whether an image benefits from a transparency key,
//! find a color not present in the image, and replace key pixels with it.
//! Port of vtracer's `frontend/keying.rs`.

use crate::error::Error;
use crate::types::{Color, ColorImage};

/// Whether `img` benefits from transparency keying.
///
/// vtracer keys images whose border is mostly one color (e.g. a white
/// matte around clip art), so the border color can be dropped to alpha.
pub fn should_key_image(img: &ColorImage) -> bool {
    if img.width == 0 || img.height == 0 {
        return false;
    }

    let w = img.width;
    let h = img.height;
    let mut border_counts: Vec<(Color, usize)> = Vec::new();

    let mut add_border = |c: Color| {
        for (bc, count) in border_counts.iter_mut() {
            if *bc == c {
                *count += 1;
                return;
            }
        }
        border_counts.push((c, 1));
    };

    // Top and bottom rows.
    for x in 0..w {
        add_border(img.get_pixel(x, 0));
        if h > 1 {
            add_border(img.get_pixel(x, h - 1));
        }
    }
    // Left and right columns (excluding corners already counted).
    for y in 1..h.saturating_sub(1) {
        add_border(img.get_pixel(0, y));
        if w > 1 {
            add_border(img.get_pixel(w - 1, y));
        }
    }

    let total_border = (2 * w + 2 * h.saturating_sub(2)).max(1);
    let mut max_count = 0usize;
    for (_, c) in &border_counts {
        if *c > max_count {
            max_count = *c;
        }
    }

    // Key when a single color dominates the border (>= 80%).
    max_count * 100 >= total_border * 80
}

/// Find a color not present anywhere in `img`.
///
/// vtracer picks a near-black color (or white if black is present) and walks
/// nearby colors until one is absent from the image.
pub fn find_unused_color(img: &ColorImage) -> Result<Color, Error> {
    let mut present: Vec<Color> = Vec::new();
    for y in 0..img.height {
        for x in 0..img.width {
            let c = img.get_pixel(x, y);
            if !present.contains(&c) {
                present.push(c);
            }
        }
    }

    // Walk nearby colors until we find one not present.
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                let c = Color::new(r, g, b);
                if !present.contains(&c) {
                    return Ok(c);
                }
            }
        }
    }

    Err(Error::Other("no unused color found in image".into()))
}

/// Replace every pixel equal to `key` with the given replacement color.
pub fn apply_key(img: &mut ColorImage, key: Color, replacement: Color) {
    for y in 0..img.height {
        for x in 0..img.width {
            if img.get_pixel(x, y) == key {
                // get_pixel returns a copy; we need mutable access.
                let idx = y * img.width + x;
                img.pixels[idx] = replacement;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_key_image_border_dominant() {
        let img = ColorImage::new(
            3,
            3,
            vec![
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
                Color::new(255, 255, 255),
            ],
        );
        assert!(should_key_image(&img));
    }

    #[test]
    fn find_unused_color_absent() {
        let img = ColorImage::new(1, 1, vec![Color::new(0, 0, 0)]);
        let c = find_unused_color(&img).unwrap();
        assert_ne!(c, Color::new(0, 0, 0));
    }

    #[test]
    fn apply_key_replaces() {
        let mut img = ColorImage::new(
            2,
            2,
            vec![
                Color::new(0, 0, 0),
                Color::new(255, 255, 255),
                Color::new(0, 0, 0),
                Color::new(128, 128, 128),
            ],
        );
        apply_key(&mut img, Color::new(0, 0, 0), Color::new(255, 0, 0));
        assert_eq!(img.get_pixel(0, 0), Color::new(255, 0, 0));
        assert_eq!(img.get_pixel(1, 0), Color::new(255, 255, 255));
        assert_eq!(img.get_pixel(0, 1), Color::new(255, 0, 0));
    }
}
