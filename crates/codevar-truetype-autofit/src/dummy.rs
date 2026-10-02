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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! Dummy writing system (`afdummy.c`).
//!
//! Used when no hinting should be performed: the outline is reloaded
//! and immediately saved back, so the glyph is left untouched.

use codevar_truetype_core::{Outline, TtResult};

use crate::hints::GlyphHints;
use crate::metrics::StyleMetrics;

/// `af_dummy_hints_init` (`afdummy.c`): attaches the metrics to the
/// hint record and copies the scaler transformation.
///
/// The dummy writing system has no `style_metrics_init` /
/// `style_metrics_scale` entry points in C; only the hint side exists.
pub fn hints_init(hints: &mut GlyphHints, metrics: &StyleMetrics) {
    hints.rescale(metrics);
    let scaler = metrics.scaler();
    hints.x_scale = scaler.x_scale;
    hints.y_scale = scaler.y_scale;
    hints.x_delta = scaler.x_delta;
    hints.y_delta = scaler.y_delta;
}

/// `af_dummy_hints_apply` (`afdummy.c`): reloads `outline` into the
/// hint record and saves it straight back, producing no hinting at all.
pub fn hints_apply(hints: &mut GlyphHints, outline: &mut Outline) -> TtResult<()> {
    hints.reload(outline);
    hints.save(outline);
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use codevar_truetype_core::Vector;

    use super::*;
    use crate::metrics::{GlobalsShared, StyleMetrics};
    use crate::ranges::{STYLE_CLASSES, Style};

    fn metrics(style: Style) -> StyleMetrics {
        StyleMetrics::new(&STYLE_CLASSES[style.index()], Rc::new(GlobalsShared::new(2048)))
    }

    fn square() -> Outline {
        let mut outline = Outline::with_capacity(4, 1);
        outline.points = vec![
            Vector::new(10, 20),
            Vector::new(110, 20),
            Vector::new(110, 220),
            Vector::new(10, 220),
        ];
        outline.tags = vec![0, 0, 0, 0];
        outline.contours = vec![3];
        outline.n_points = 4;
        outline.n_contours = 1;
        outline
    }

    #[test]
    fn hints_init_copies_the_scaler() {
        let mut metrics = metrics(Style::NoneDflt);
        metrics.scaler_mut().x_scale = 3 << 16;
        metrics.scaler_mut().y_scale = 5 << 16;
        metrics.scaler_mut().x_delta = 7;
        metrics.scaler_mut().y_delta = 11;

        let mut hints = GlyphHints::default();
        hints_init(&mut hints, &metrics);

        assert_eq!(hints.x_scale, 3 << 16);
        assert_eq!(hints.y_scale, 5 << 16);
        assert_eq!(hints.x_delta, 7);
        assert_eq!(hints.y_delta, 11);
        assert!(hints.metrics.is_some());
    }

    #[test]
    fn hints_apply_leaves_the_outline_untouched() {
        let before = square();
        let mut outline = before.clone();

        let mut metrics = metrics(Style::NoneDflt);
        metrics.scaler_mut().x_scale = 1 << 16;
        metrics.scaler_mut().y_scale = 1 << 16;

        let mut hints = GlyphHints::default();
        hints_init(&mut hints, &metrics);
        hints_apply(&mut hints, &mut outline).expect("reload cannot fail");

        assert_eq!(outline, before);
        assert_eq!(hints.num_points(), 4);
        assert_eq!(hints.num_contours(), 1);
    }
}
