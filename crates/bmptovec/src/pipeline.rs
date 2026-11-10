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

//! Pipeline driver: frontend → color fitters → compositing → curve passes →
//! optimizers → SVG writer.
//!
//! Faithful port of visioncortex/vtracer's `crates/vtracer/src/pipeline.rs`.

use crate::VectorDoc;
use crate::colorfit::ColorFitter;
use crate::compose::Compositing;
use crate::error::Error;
use crate::frontend::Frontend;
use crate::ir::Segmentation;
use crate::optimize::OptimizerPass;
use crate::progress::{CancelToken, Ctx};
use crate::simplify::CurvePass;
use crate::svg::SvgWriter;
use crate::types::ColorImage;

/// The full raster-to-vector pipeline, assembled from a [`Config`].
pub struct Pipeline {
    frontend: Box<dyn Frontend>,
    color_fitters: Vec<Box<dyn ColorFitter>>,
    compositing: Compositing,
    curve_passes: Vec<Box<dyn CurvePass>>,
    optimizers: Vec<Box<dyn OptimizerPass>>,
    writer: SvgWriter,
}

impl Pipeline {
    /// Assemble a pipeline from its stages.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frontend: Box<dyn Frontend>,
        color_fitters: Vec<Box<dyn ColorFitter>>,
        compositing: Compositing,
        curve_passes: Vec<Box<dyn CurvePass>>,
        optimizers: Vec<Box<dyn OptimizerPass>>,
        writer: SvgWriter,
    ) -> Self {
        Self {
            frontend,
            color_fitters,
            compositing,
            curve_passes,
            optimizers,
            writer,
        }
    }

    /// Borrow the frontend.
    #[inline]
    pub fn frontend(&self) -> &dyn Frontend {
        self.frontend.as_ref()
    }

    /// Borrow the optimizer passes.
    #[inline]
    pub fn optimizers(&self) -> &[Box<dyn OptimizerPass>] {
        &self.optimizers
    }

    /// Borrow the SVG writer.
    #[inline]
    pub fn writer(&self) -> &SvgWriter {
        &self.writer
    }

    /// Segment `img` into a [`Segmentation`] (frontend stage only).
    pub fn segment(&self, img: &ColorImage) -> Result<Segmentation, Error> {
        let token = CancelToken::new();
        let mut ctx = Ctx::new(&token);
        self.frontend.segment_with(img, &mut ctx)
    }

    /// Run the full fitting/composing pipeline on a segmentation, producing a
    /// [`VectorDoc`].
    pub fn finish(&self, seg: &Segmentation) -> Result<VectorDoc, Error> {
        // Apply color fitters to the segmentation (in place).
        let mut seg = seg.clone();
        for fitter in &self.color_fitters {
            fitter.fit(&mut seg);
        }

        // Fit each layer into vector paths via the segment fitter / compositing.
        // Curve simplification passes are applied inside `compositing.compose`
        // (per-contour `.ring()`) so shared mosaic boundaries stay seam-free.
        let doc = self.compositing.compose(&seg, &self.curve_passes);

        Ok(doc)
    }

    /// Convert an image straight to an SVG string: segment → fitters →
    /// compose → optimizers → write.
    pub fn to_svg(&self, img: &ColorImage) -> Result<String, Error> {
        let seg = self.segment(img)?;
        let mut doc = self.finish(&seg)?;
        for pass in &self.optimizers {
            pass.run(&mut doc);
        }
        Ok(self.writer.write(&doc))
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Clustering, Config, FitMode, Hierarchical, Preset};

    #[test]
    fn segment_key_matches_default_config() {
        let cfg = Config::default();
        assert_eq!(cfg.clustering, Clustering::Binary);
        assert_eq!(cfg.hierarchical, Hierarchical::Stacked);
        assert_eq!(cfg.fit_mode, FitMode::Pixel);
        assert_eq!(cfg.preset, Preset::Bw);
    }

    #[test]
    fn config_builds_pipeline() {
        let cfg = Config::from_preset(Preset::Poster);
        let _pipe = cfg.build().expect("poster config builds");
    }
}
