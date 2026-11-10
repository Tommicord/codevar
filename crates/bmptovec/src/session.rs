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

//! Cached pipeline session: holds a [`Config`] and [`Pipeline`], memoizing the
//! last segmentation so repeated conversions of the same image/key skip
//! re-segmentation.
//!
//! Faithful port of visioncortex/vtracer's `crates/vtracer/src/session.rs`.

use crate::VectorDoc;
use crate::config::{Config, SegmentKey};
use crate::error::Error;
use crate::ir::Segmentation;
use crate::pipeline::Pipeline;
use crate::progress::{CancelToken, Ctx};
use crate::types::ColorImage;

/// A cached pipeline session.
///
/// Re-segmentation only happens when the segment key (canvas size + clustering
/// settings) changes; otherwise the cached segmentation is reused.
pub struct Session {
    cfg: Config,
    pipeline: Pipeline,
    cached_seg: Option<Segmentation>,
    cached_key: Option<SegmentKey>,
}

impl Session {
    /// Create a new session from a config.
    pub fn new(cfg: Config) -> Result<Self, Error> {
        let pipeline = cfg.build()?;
        Ok(Self {
            cfg,
            pipeline,
            cached_seg: None,
            cached_key: None,
        })
    }

    /// Borrow the underlying config.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Borrow the underlying pipeline.
    #[inline]
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Segment `img`, reusing the cached segmentation when the key is
    /// unchanged.
    pub fn segment(&mut self, img: &ColorImage) -> Result<&Segmentation, Error> {
        let key = self
            .cfg
            .segment_key(img.width as u32, img.height as u32);
        if self.cached_key.as_ref() != Some(&key) {
            let token = CancelToken::new();
            let mut ctx = Ctx::new(&token);
            let seg = self
                .pipeline
                .frontend()
                .segment_with(img, &mut ctx)?;
            self.cached_key = Some(key);
            self.cached_seg = Some(seg);
        }
        let seg = self
            .cached_seg
            .as_ref()
            .ok_or(Error::Other("segmentation missing after caching".to_string()))?;
        Ok(seg)
    }

    /// Run the full fitting/composing pipeline on a segmentation, producing a
    /// [`VectorDoc`].
    pub fn finish(&self, seg: &Segmentation) -> Result<VectorDoc, Error> {
        self.pipeline.finish(seg)
    }

    /// Convert an image straight to an SVG string, using the cached
    /// segmentation when possible.
    pub fn to_svg(&mut self, img: &ColorImage) -> Result<String, Error> {
        let seg = self.segment(img)?.clone();
        let mut doc = self.pipeline.finish(&seg)?;
        for pass in self.pipeline.optimizers() {
            pass.run(&mut doc);
        }
        Ok(self.pipeline.writer().write(&doc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;

    #[test]
    fn session_new_builds() {
        let cfg = Config::from_preset(Preset::Bw);
        let _sess = Session::new(cfg).expect("session builds");
    }
}
