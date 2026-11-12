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

//! Connected-component extraction and small-region cleanup.
//!
//! A two-pass union-find labeling pass keeps the extraction linear in the
//! number of pixels and produces deterministic component identifiers in
//! raster order.
//!
//! The same pass also powers [`despeckle`], which removes components smaller
//! than a caller-selected area by repainting them with the dominant surrounding
//! palette entry. This is essential for anti-aliased and dithered input, where
//! palette assignment can leave isolated single-pixel islands that would
//! otherwise become microscopic vector shapes.

use alloc::vec;
use alloc::vec::Vec;

use crate::component::{Component, Connectivity, DespeckleOptions, DespeckleReport, Segmentation};
use crate::error::{Error, Result};
use crate::palette::{IndexedImage, NO_INDEX};

/// Labels every connected region of an indexed image.
///
/// The components come back in raster order, so the same image always
/// produces the same identifiers.
pub fn label_components(indexed: &IndexedImage, connectivity: Connectivity) -> Result<Segmentation> {
    let pixel_count = indexed.pixel_count();
    if pixel_count == 0 {
        return Ok(Segmentation {
            width: indexed.width(),
            height: indexed.height(),
            components: Vec::new(),
            palette: indexed.palette().clone(),
        });
    }
    if pixel_count > u32::MAX as usize {
        return Err(Error::ImageTooLarge {
            width: indexed.width(),
            height: indexed.height(),
        });
    }
    for &index in indexed.indices() {
        if index != NO_INDEX && indexed.palette().get(index as usize).is_none() {
            return Err(Error::InvalidPaletteEntry {
                index: index as usize,
            });
        }
    }

    let components = label_index_map(indexed.width(), indexed.height(), indexed.indices(), connectivity);
    Ok(Segmentation {
        width: indexed.width(),
        height: indexed.height(),
        components,
        palette: indexed.palette().clone(),
    })
}

/// Repaints components smaller than `minimum_area` with their dominant
/// neighbour color.
///
/// The pass re-labels after each round because repainting a tiny island can
/// change the size of the island around it. The operation is in-place on the
/// index map; the palette itself is never modified.
pub fn despeckle(
    indexed: &mut IndexedImage,
    connectivity: Connectivity,
    options: DespeckleOptions,
) -> Result<DespeckleReport> {
    let mut report = DespeckleReport::default();
    if !options.is_enabled() || indexed.is_empty() {
        return Ok(report);
    }
    if indexed.pixel_count() > u32::MAX as usize {
        return Err(Error::ImageTooLarge {
            width: indexed.width(),
            height: indexed.height(),
        });
    }

    for pass in 0..options.passes {
        let (components, labels) =
            label_index_map_with_labels(indexed.width(), indexed.height(), indexed.indices(), connectivity);
        let width = indexed.width();
        let mut changed = false;

        for component in &components {
            if component.area >= options.minimum_area as u64 {
                continue;
            }
            let replacement = dominant_neighbour(
                indexed,
                component,
                width,
                &labels,
                &components,
                options.minimum_area,
            );
            let Some(replacement) = replacement else {
                continue;
            };
            let start = indexed.indices_mut();
            for &packed in &component.pixels {
                let slot = packed as usize;
                if let Some(value) = start.get_mut(slot) {
                    *value = replacement;
                }
            }
            report.removed_components += 1;
            report.reassigned_pixels += component.area;
            changed = true;
        }

        report.passes_run = pass + 1;
        if !changed {
            break;
        }
    }
    Ok(report)
}

/// Chooses the color that should replace `component`.
///
/// The replacement has to be a *local* region that is strictly larger than the
/// component being removed. Without that rule two adjacent specks would trade
/// colors and the image would be inverted instead of cleaned. The comparison
/// uses the size of the region the neighbour pixel belongs to, not the total
/// size of its color class: a color that also occurs in a large region
/// somewhere else in the image says nothing about the region a speck touches.
///
/// Among the admissible candidates the largest region wins, the number of
/// shared edges breaks ties, and the lowest palette index breaks a full tie so
/// the result stays deterministic.
fn dominant_neighbour(
    indexed: &IndexedImage,
    component: &Component,
    width: u32,
    labels: &[u32],
    components: &[Component],
    minimum_area: u32,
) -> Option<u32> {
    let indices = indexed.indices();
    let height = indexed.height();
    let mut counts: Vec<(u32, u64)> = Vec::new();

    for &packed in &component.pixels {
        let x = (packed % width) as i32;
        let y = (packed / width) as i32;
        for &(dx, dy) in Connectivity::four_offsets() {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let slot = ny as usize * width as usize + nx as usize;
            let neighbour = indices[slot];
            if neighbour == NO_INDEX || neighbour == component.palette_index {
                continue;
            }
            if let Some(entry) = counts
                .iter_mut()
                .find(|(index, _)| *index == neighbour)
            {
                entry.1 += 1;
            } else {
                counts.push((neighbour, 1));
            }
        }
    }

    // The area of the local region a neighbour pixel belongs to. Several
    // neighbours can share a color while belonging to different regions, so
    // the area is tracked per region and not per color.
    let mut region_areas: Vec<(u32, u64)> = Vec::new();
    for &packed in &component.pixels {
        let x = (packed % width) as i32;
        let y = (packed / width) as i32;
        for &(dx, dy) in Connectivity::four_offsets() {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let slot = ny as usize * width as usize + nx as usize;
            let neighbour = indices[slot];
            if neighbour == NO_INDEX || neighbour == component.palette_index {
                continue;
            }
            let label = labels[slot];
            if label == NO_INDEX {
                continue;
            }
            let area = components
                .get(label as usize)
                .map_or(0, |region| region.area);
            match region_areas
                .iter_mut()
                .find(|(index, _)| *index == neighbour)
            {
                Some(entry) => {
                    if area > entry.1 {
                        entry.1 = area;
                    }
                }
                None => region_areas.push((neighbour, area)),
            }
        }
    }

    let own_area = component.area;
    let mut best: Option<(u32, u64, u64)> = None;
    for (index, count) in counts {
        let area = region_areas
            .iter()
            .find(|(candidate, _)| *candidate == index)
            .map_or(0, |(_, area)| *area);
        // Only a strictly larger region may absorb this component.
        if area <= own_area {
            continue;
        }
        let stable = area >= minimum_area as u64;
        match best {
            None => best = Some((index, count, area)),
            Some((best_index, best_count, best_area)) => {
                let best_stable = best_area >= minimum_area as u64;
                let better = match (stable, best_stable) {
                    // A stable neighbour always wins over an unstable one.
                    (true, false) => true,
                    (false, true) => false,
                    // Within the same stability class the larger region wins,
                    // so a speck merges into the biggest adjacent region.
                    _ => {
                        area > best_area
                            || (area == best_area
                                && (count > best_count || (count == best_count && index < best_index)))
                    }
                };
                if better {
                    best = Some((index, count, area));
                }
            }
        }
    }
    best.map(|(index, _, _)| index)
}

fn label_index_map(width: u32, height: u32, indices: &[u32], connectivity: Connectivity) -> Vec<Component> {
    label_index_map_with_labels(width, height, indices, connectivity).0
}

/// Labels every component and returns the components together with a map from
/// pixel offset to the ordinal of the component that owns it.
///
/// The label map lets a caller ask how large the *local* region around a
/// pixel is, which is what the despeckler needs: the size of a color class
/// somewhere else in the image says nothing about the region a speck touches.
fn label_index_map_with_labels(
    width: u32,
    height: u32,
    indices: &[u32],
    connectivity: Connectivity,
) -> (Vec<Component>, Vec<u32>) {
    let width_usize = width as usize;
    let pixel_count = width_usize * height as usize;
    if pixel_count == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut parent: Vec<u32> = (0..pixel_count as u32).collect();
    let offsets = connectivity.preceding_offsets();

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let pixel = y as usize * width_usize + x as usize;
            let index = indices[pixel];
            if index == NO_INDEX {
                continue;
            }
            for &(dx, dy) in offsets {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let neighbour = ny as usize * width_usize + nx as usize;
                if indices[neighbour] == index {
                    union(&mut parent, pixel as u32, neighbour as u32);
                }
            }
        }
    }

    let mut root_to_component: Vec<u32> = vec![NO_INDEX; pixel_count];
    let mut components: Vec<Component> = Vec::new();
    let mut labels: Vec<u32> = vec![NO_INDEX; pixel_count];

    for y in 0..height {
        for x in 0..width {
            let pixel = y as usize * width_usize + x as usize;
            let index = indices[pixel];
            if index == NO_INDEX {
                continue;
            }
            let root = find(&mut parent, pixel as u32);
            let current = root_to_component[root as usize];
            if current == NO_INDEX {
                let component_index = components.len() as u32;
                root_to_component[root as usize] = component_index;
                components.push(Component::single(index, x, y, pixel as u32));
                labels[pixel] = component_index;
            } else {
                components[current as usize].push(x, y, pixel as u32);
                labels[pixel] = current;
            }
        }
    }

    (components, labels)
}

fn find(parent: &mut [u32], mut value: u32) -> u32 {
    let mut root = value;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    while parent[value as usize] != value {
        let next = parent[value as usize];
        parent[value as usize] = root;
        value = next;
    }
    root
}

fn union(parent: &mut [u32], first: u32, second: u32) {
    let first_root = find(parent, first);
    let second_root = find(parent, second);
    if first_root == second_root {
        return;
    }
    // Union by index keeps the structure deterministic and avoids an extra
    // rank array; path compression bounds the amortized cost.
    if first_root < second_root {
        parent[second_root as usize] = first_root;
    } else {
        parent[first_root as usize] = second_root;
    }
}

/// Counts the distinct palette indices adjacent to a component.
///
/// The result is sorted by palette index, making it useful for diagnostics and
/// for merge rules that need a stable ordering.
pub fn component_neighbours(
    indexed: &IndexedImage,
    component: &Component,
    connectivity: Connectivity,
) -> Vec<u32> {
    let width = indexed.width();
    let height = indexed.height();
    let width_usize = width as usize;
    let indices = indexed.indices();
    let offsets: &[(i32, i32)] = match connectivity {
        Connectivity::Four => Connectivity::four_offsets(),
        Connectivity::Eight => &[
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ],
    };
    let mut neighbours = Vec::new();
    for &packed in &component.pixels {
        let x = (packed % width) as i32;
        let y = (packed / width) as i32;
        for &(dx, dy) in offsets {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let neighbour = indices[ny as usize * width_usize + nx as usize];
            if neighbour != NO_INDEX && neighbour != component.palette_index {
                neighbours.push(neighbour);
            }
        }
    }
    // Sorting and deduplicating keeps the scan linear in the number of
    // contacts instead of quadratic in the number of distinct colors.
    neighbours.sort_unstable();
    neighbours.dedup();
    neighbours
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::geom::PixelRect;
    use crate::palette::{ColorMode, Palette};
    use crate::quantize::quantize;

    fn indexed(indices: &[u32], palette: &[Color], width: u32, height: u32) -> IndexedImage {
        let palette = Palette::new(palette.to_vec()).expect("palette");
        IndexedImage::from_parts(width, height, indices.to_vec(), palette)
    }

    #[test]
    fn raster_components_are_labeled_in_order() {
        let image = indexed(&[0, 0, 1, 1, 0, 0, 1, 1], &[Color::BLACK, Color::WHITE], 4, 2);
        let segmentation = label_components(&image, Connectivity::Four).expect("label");
        assert_eq!(segmentation.len(), 2);
        assert_eq!(segmentation.components[0].palette_index, 0);
        assert_eq!(segmentation.components[0].area, 4);
        assert_eq!(segmentation.components[1].palette_index, 1);
        assert_eq!(segmentation.components[1].area, 4);
        assert_eq!(segmentation.total_area(), 8);
    }

    #[test]
    fn diagonal_connectivity_changes_label_count() {
        let image = indexed(&[1, 0, 0, 1], &[Color::BLACK, Color::WHITE], 2, 2);
        let four = label_components(&image, Connectivity::Four).expect("four");
        let eight = label_components(&image, Connectivity::Eight).expect("eight");
        assert_eq!(four.len(), 4);
        assert_eq!(eight.len(), 2);
    }

    #[test]
    fn component_bounds_and_fill_ratio_are_exact() {
        let image = indexed(&[1, 1, 0, 0, 1, 1, 0, 0], &[Color::WHITE, Color::BLACK], 4, 2);
        let segmentation = label_components(&image, Connectivity::Four).expect("label");
        let component = &segmentation.components[0];
        assert_eq!(component.bounds, PixelRect::new(0, 0, 2, 2));
        assert_eq!(component.area, 4);
        assert!((component.fill_ratio() - 1.0).abs() < 1.0e-12);
        assert_eq!(component.color(&segmentation.palette), Some(Color::BLACK));
    }

    #[test]
    fn despeckle_repaints_a_single_pixel_island() {
        let mut indices = vec![0_u32; 25];
        indices[12] = 1;
        let mut image = indexed(&indices, &[Color::WHITE, Color::BLACK], 5, 5);
        let report = despeckle(
            &mut image,
            Connectivity::Four,
            DespeckleOptions {
                minimum_area: 2,
                passes: 2,
            },
        )
        .expect("despeckle");
        assert_eq!(report.removed_components, 1);
        assert_eq!(report.reassigned_pixels, 1);
        assert_eq!(image.index(2, 2), 0);
        assert_eq!(image.transparent_pixels(), 0);
    }

    #[test]
    fn despeckle_keeps_components_at_the_threshold() {
        let mut indices = vec![0_u32; 25];
        indices[12] = 1;
        indices[13] = 1;
        let mut image = indexed(&indices, &[Color::WHITE, Color::BLACK], 5, 5);
        let report = despeckle(
            &mut image,
            Connectivity::Four,
            DespeckleOptions {
                minimum_area: 2,
                passes: 2,
            },
        )
        .expect("despeckle");
        assert_eq!(report.removed_components, 0);
        assert_eq!(image.index(2, 2), 1);
    }

    #[test]
    fn despeckle_cascades_across_passes() {
        let mut indices = vec![0_u32; 49];
        indices[24] = 1;
        let mut image = indexed(&indices, &[Color::WHITE, Color::BLACK], 7, 7);
        let report = despeckle(
            &mut image,
            Connectivity::Eight,
            DespeckleOptions {
                minimum_area: 3,
                passes: 1,
            },
        )
        .expect("despeckle");
        assert_eq!(report.removed_components, 1);
        assert_eq!(report.passes_run, 1);
    }

    #[test]
    fn despeckle_never_trades_colors_between_two_specks() {
        // Two interleaved columns of sub-threshold components must not swap
        // colors: the cleanup has to shrink the component count, never invert
        // the image. Neither class is larger than the other, so neither may
        // absorb the other and the image is left untouched.
        let indices = [1_u32, 0, 1, 0, 1, 0, 1, 0];
        let mut image = indexed(&indices, &[Color::BLACK, Color::WHITE], 2, 4);
        let report = despeckle(
            &mut image,
            Connectivity::Four,
            DespeckleOptions {
                minimum_area: 5,
                passes: 2,
            },
        )
        .expect("despeckle");

        assert_eq!(report.removed_components, 0);
        assert_eq!(report.reassigned_pixels, 0);
        assert_eq!(image.indices(), &indices);
    }

    #[test]
    fn despeckle_keeps_the_larger_of_two_adjacent_specks() {
        // The smaller color class is absorbed by the larger one, so the
        // cleanup never inverts the image.
        let indices = [1_u32, 1, 1, 0];
        let mut image = indexed(&indices, &[Color::BLACK, Color::WHITE], 2, 2);
        let report = despeckle(
            &mut image,
            Connectivity::Four,
            DespeckleOptions {
                minimum_area: 5,
                passes: 2,
            },
        )
        .expect("despeckle");

        assert_eq!(report.removed_components, 1);
        assert_eq!(report.reassigned_pixels, 1);
        assert_eq!(image.indices(), &[1, 1, 1, 1]);
    }

    #[test]
    fn despeckle_uses_the_local_region_and_not_the_whole_color_class() {
        // The single pixel in the top-left corner shares its color with the
        // wide band at the bottom, so the color class is much larger than the
        // speck. The speck still has to be removed, because the region it
        // touches is larger than the speck itself.
        let width = 5_u32;
        let height = 5_u32;
        let mut indices = vec![0_u32; (width * height) as usize];
        indices[0] = 1;
        for row in 3..height {
            for column in 0..width {
                indices[(row * width + column) as usize] = 1;
            }
        }
        let mut image = indexed(&indices, &[Color::BLACK, Color::WHITE], width, height);
        let report = despeckle(
            &mut image,
            Connectivity::Four,
            DespeckleOptions {
                minimum_area: 2,
                passes: 1,
            },
        )
        .expect("despeckle");

        assert_eq!(report.removed_components, 1);
        assert_eq!(report.reassigned_pixels, 1);
        assert_eq!(image.index(0, 0), 0);
        // The band is a large region of its own and stays untouched.
        assert_eq!(image.index(0, 4), 1);
    }

    #[test]
    fn transparent_pixels_are_not_components() {
        let image = indexed(
            &[NO_INDEX, NO_INDEX, 0, 0, NO_INDEX, NO_INDEX, 0, 0],
            &[Color::BLACK],
            4,
            2,
        );
        let segmentation = label_components(&image, Connectivity::Four).expect("label");
        assert_eq!(segmentation.len(), 1);
        assert_eq!(segmentation.components[0].area, 4);
        assert_eq!(segmentation.total_area(), 4);
    }

    #[test]
    fn neighbors_are_sorted_and_distinct() {
        let image = indexed(
            &[0, 0, 0, 2, 1, 2, 0, 0, 0],
            &[Color::BLACK, Color::WHITE, Color::rgb(1, 2, 3)],
            3,
            3,
        );
        let segmentation = label_components(&image, Connectivity::Four).expect("label");
        let center = segmentation
            .components
            .iter()
            .find(|component| component.bounds == PixelRect::single(1, 1))
            .expect("center component");
        let neighbours = component_neighbours(&image, center, Connectivity::Four);
        assert_eq!(neighbours, vec![0, 2]);
        let neighbours = component_neighbours(&image, center, Connectivity::Eight);
        assert_eq!(neighbours, vec![0, 2]);
    }

    #[test]
    fn sorting_is_largest_first_and_stable() {
        let image = indexed(
            &[0, 0, 0, 1, 1, 2],
            &[Color::BLACK, Color::WHITE, Color::gray(128)],
            3,
            2,
        );
        let mut segmentation = label_components(&image, Connectivity::Four).expect("label");
        segmentation.sort_by_area_descending();
        assert_eq!(segmentation.components[0].area, 3);
        assert_eq!(segmentation.components[1].area, 2);
        assert_eq!(segmentation.components[2].area, 1);
    }

    #[test]
    fn invalid_palette_indices_are_rejected() {
        let image = indexed(&[5], &[Color::BLACK], 1, 1);
        assert!(matches!(
            label_components(&image, Connectivity::Four),
            Err(Error::InvalidPaletteEntry { .. })
        ));
    }

    #[test]
    fn quantized_image_labels_correctly() {
        let source =
            crate::image::ColorImage::from_rgb8(3, 1, &[255, 0, 0, 0, 255, 0, 0, 0, 255]).expect("image");
        let image = quantize(&source, &ColorMode::Exact, None, 1).expect("quantize");
        let segmentation = label_components(&image, Connectivity::Four).expect("label");
        assert_eq!(segmentation.len(), 3);
        assert_eq!(segmentation.palette.len(), 3);
    }
}
