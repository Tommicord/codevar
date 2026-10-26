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

//! Style metrics (`AF_StyleMetricsRec` and its writing system
//! derivations, `aftypes.h`).
//!
//! In FreeType every writing system allocates its own metrics
//! structure (for example `AF_LatinMetricsRec`) whose first member is
//! the common `AF_StyleMetricsRec`, and the auto-hinter downcasts the
//! base pointer whenever it needs the derived fields.  This port keeps
//! the same layout with a tagged [`StyleMetrics`] enum instead of an
//! unsafe cast.

use alloc::rc::Rc;
use core::cell::Cell;

use crate::Scaler;
use crate::cjk::CjkMetrics;
use crate::latin::LatinMetrics;
use crate::ranges::{Style, StyleClass, WritingSystem};

/// `AF_PROP_INCREASE_X_HEIGHT_MIN` (`afglobal.h`): smallest pixel
/// size for which the x-height increase is applied.
pub const PROP_INCREASE_X_HEIGHT_MIN: u32 = 6;
/// `AF_PROP_INCREASE_X_HEIGHT_MAX` (`afglobal.h`): default (disabled)
/// value of the `increase-x-height` property.
pub const PROP_INCREASE_X_HEIGHT_MAX: u32 = 0;

/// Face wide data shared between [`FaceGlobals`](crate::FaceGlobals)
/// and every [`StyleMetrics`] of the face (`AF_FaceGlobalsRec` fields
/// read through `metrics->globals` in C).
///
/// FreeType stores a back pointer to `AF_FaceGlobalsRec`; since the
/// metrics live inside the globals structure this port shares a small
/// `Rc` cell instead, which keeps the same aliasing behavior without
/// unsafe code.
#[derive(Debug)]
pub struct GlobalsShared {
    /// `face->units_per_EM`.
    pub units_per_em: u16,
    /// The `increase-x-height` property (`AF_PROP_INCREASE_X_HEIGHT_*`).
    increase_x_height: Cell<u32>,
    /// The `warping` property of the auto-hinter module, reached in C
    /// through `metrics->globals->module->warping`.
    warping: Cell<bool>,
}

impl GlobalsShared {
    /// Creates the shared cell for a face with the given
    /// `units_per_EM`.
    #[inline]
    pub fn new(units_per_em: u16) -> GlobalsShared {
        GlobalsShared {
            units_per_em,
            increase_x_height: Cell::new(PROP_INCREASE_X_HEIGHT_MAX),
            warping: Cell::new(false),
        }
    }

    /// Current value of the `increase-x-height` property.
    #[inline]
    pub fn increase_x_height(&self) -> u32 {
        self.increase_x_height.get()
    }

    /// Updates the `increase-x-height` property (values outside
    /// \[`PROP_INCREASE_X_HEIGHT_MIN` .. `PROP_INCREASE_X_HEIGHT_MAX`\]
    /// are clamped by the property service, as in `afmodule.c`).
    #[inline]
    pub fn set_increase_x_height(&self, value: u32) {
        self.increase_x_height.set(value);
    }

    /// Current value of the module wide `warping` property
    /// (`AF_ModuleRec::warping`, initialized to `0` by
    /// `af_autofitter_init`).
    #[inline]
    pub fn warping(&self) -> bool {
        self.warping.get()
    }

    /// Updates the module wide `warping` property.
    #[inline]
    pub fn set_warping(&self, value: bool) {
        self.warping.set(value);
    }
}

/// `AF_StyleMetricsRec` (`aftypes.h`): the part of the global metrics
/// common to every writing system.
#[derive(Clone, Debug)]
pub struct StyleMetricsRec {
    /// The style class this metrics object was created for.
    pub style_class: &'static StyleClass,
    /// The scaler of the target size (updated for each glyph).
    pub scaler: Scaler,
    /// True if all digits of the face have the same width
    /// (`AF_StyleMetricsRec::digits_have_same_width`).
    pub digits_have_same_width: bool,
    /// Face wide shared data (`AF_StyleMetricsRec::globals`).
    pub globals: Rc<GlobalsShared>,
}

impl StyleMetricsRec {
    /// Creates a base record for the given style and face.
    #[inline]
    pub fn new(style_class: &'static StyleClass, globals: Rc<GlobalsShared>) -> StyleMetricsRec {
        StyleMetricsRec {
            style_class,
            scaler: Scaler::default(),
            digits_have_same_width: false,
            globals,
        }
    }
}

/// The per-writing-system global metrics of one style
/// (`AF_StyleMetrics` and its derivations).
///
/// * [`StyleMetrics::Dummy`] is a plain `AF_StyleMetricsRec`
///   (`afdummy.c` never allocates extra fields),
/// * [`StyleMetrics::Latin`] is `AF_LatinMetricsRec` (`aflatin.h`,
///   shared by `aflatin2.c`),
/// * [`StyleMetrics::Cjk`] is `AF_CJKMetricsRec` (`afcjk.h`), also
///   used by the Indic writing system (`afindic.c`).
#[derive(Clone, Debug)]
pub enum StyleMetrics {
    /// Metrics of the dummy writing system.
    Dummy(StyleMetricsRec),
    /// Metrics of the Latin writing system.
    Latin(LatinMetrics),
    /// Metrics of the CJK (and Indic) writing system.
    Cjk(CjkMetrics),
}

impl StyleMetrics {
    /// Creates the metrics variant matching `writing_system` for a
    /// style (mirrors `writing_system_class->style_metrics_size` plus
    /// the `metrics->style_class` / `metrics->globals` assignment of
    /// `af_face_globals_get_metrics`).
    pub fn new(style_class: &'static StyleClass, globals: Rc<GlobalsShared>) -> StyleMetrics {
        let root = StyleMetricsRec::new(style_class, globals);
        match style_class.writing_system {
            WritingSystem::Dummy => StyleMetrics::Dummy(root),
            WritingSystem::Latin => StyleMetrics::Latin(LatinMetrics {
                root,
                units_per_em: 0,
                axis: Default::default(),
            }),
            WritingSystem::Cjk | WritingSystem::Indic => StyleMetrics::Cjk(CjkMetrics {
                root,
                units_per_em: 0,
                axis: Default::default(),
            }),
        }
    }

    /// The common part of the metrics (`(AF_StyleMetrics)metrics` in
    /// C).
    #[inline]
    pub fn root(&self) -> &StyleMetricsRec {
        match self {
            StyleMetrics::Dummy(root) => root,
            StyleMetrics::Latin(m) => &m.root,
            StyleMetrics::Cjk(m) => &m.root,
        }
    }

    /// Mutable access to the common part of the metrics.
    #[inline]
    pub fn root_mut(&mut self) -> &mut StyleMetricsRec {
        match self {
            StyleMetrics::Dummy(root) => root,
            StyleMetrics::Latin(m) => &mut m.root,
            StyleMetrics::Cjk(m) => &mut m.root,
        }
    }

    /// The style class of this metrics object.
    #[inline]
    pub fn style_class(&self) -> &'static StyleClass {
        self.root().style_class
    }

    /// The `AF_Style` of this metrics object.
    #[inline]
    pub fn style(&self) -> Style {
        self.root().style_class.style
    }

    /// The scaler of the target size.
    #[inline]
    pub fn scaler(&self) -> &Scaler {
        &self.root().scaler
    }

    /// Mutable access to the scaler of the target size.
    #[inline]
    pub fn scaler_mut(&mut self) -> &mut Scaler {
        &mut self.root_mut().scaler
    }

    /// `face->units_per_EM`, as read through `hints->metrics`.
    #[inline]
    pub fn units_per_em_face(&self) -> u16 {
        self.root().globals.units_per_em
    }

    /// The Latin metrics, if this is a Latin style.
    #[inline]
    pub fn latin(&self) -> Option<&LatinMetrics> {
        match self {
            StyleMetrics::Latin(m) => Some(m),
            _ => None,
        }
    }

    /// The CJK metrics, if this is a CJK or Indic style.
    #[inline]
    pub fn cjk(&self) -> Option<&CjkMetrics> {
        match self {
            StyleMetrics::Cjk(m) => Some(m),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranges::{STYLE_CLASSES, Style};

    fn globals(units_per_em: u16) -> Rc<GlobalsShared> {
        Rc::new(GlobalsShared::new(units_per_em))
    }

    fn metrics_for(style: Style, units_per_em: u16) -> StyleMetrics {
        StyleMetrics::new(&STYLE_CLASSES[style.index()], globals(units_per_em))
    }

    #[test]
    fn globals_shared_carries_upem_and_property() {
        let shared = GlobalsShared::new(2048);
        assert_eq!(shared.units_per_em, 2048);
        assert_eq!(shared.increase_x_height(), PROP_INCREASE_X_HEIGHT_MAX);
        shared.set_increase_x_height(12);
        assert_eq!(shared.increase_x_height(), 12);
        shared.set_increase_x_height(0);
        assert_eq!(shared.increase_x_height(), 0);
    }

    #[test]
    fn property_bounds_match_afglobal() {
        assert_eq!(PROP_INCREASE_X_HEIGHT_MIN, 6);
        assert_eq!(PROP_INCREASE_X_HEIGHT_MAX, 0);
    }

    #[test]
    fn new_variant_matches_the_writing_system() {
        assert!(matches!(
            metrics_for(Style::NoneDflt, 1000),
            StyleMetrics::Dummy(_)
        ));
        assert!(matches!(
            metrics_for(Style::LatnDflt, 1000),
            StyleMetrics::Latin(_)
        ));
        assert!(matches!(metrics_for(Style::HaniDflt, 1000), StyleMetrics::Cjk(_)));
        assert!(
            matches!(metrics_for(Style::BengDflt, 1000), StyleMetrics::Cjk(_)),
            "Indic delegates to the CJK metrics"
        );
    }

    #[test]
    fn downcast_accessors_follow_the_variant() {
        let latin = metrics_for(Style::LatnDflt, 1000);
        assert!(latin.latin().is_some());
        assert!(latin.cjk().is_none());

        let cjk = metrics_for(Style::HaniDflt, 1000);
        assert!(cjk.cjk().is_some());
        assert!(cjk.latin().is_none());

        let indic = metrics_for(Style::BengDflt, 1000);
        assert!(indic.cjk().is_some());
        assert!(indic.latin().is_none());

        let dummy = metrics_for(Style::NoneDflt, 1000);
        assert!(dummy.latin().is_none());
        assert!(dummy.cjk().is_none());
    }

    #[test]
    fn accessors_reach_into_the_shared_root() {
        let mut metrics = metrics_for(Style::LatnDflt, 1000);
        assert_eq!(metrics.style(), Style::LatnDflt);
        assert_eq!(metrics.style_class().name, "latn_dflt");
        assert_eq!(metrics.units_per_em_face(), 1000);
        assert!(!metrics.root().digits_have_same_width);

        metrics.scaler_mut().x_scale = 1 << 16;
        assert_eq!(metrics.scaler().x_scale, 1 << 16);

        metrics.root_mut().digits_have_same_width = true;
        assert!(metrics.root().digits_have_same_width);
        assert_eq!(metrics.style_class().style, Style::LatnDflt);
    }

    #[test]
    fn base_record_starts_with_defaults() {
        let record = StyleMetricsRec::new(&STYLE_CLASSES[Style::LatnDflt.index()], globals(2048));
        assert_eq!(record.scaler, Scaler::default());
        assert!(!record.digits_have_same_width);
        assert_eq!(record.globals.units_per_em, 2048);
        assert_eq!(record.style_class.style, Style::LatnDflt);

        let cloned = record.clone();
        assert_eq!(cloned.style_class.style, Style::LatnDflt);
        assert_eq!(cloned.globals.units_per_em, 2048);
    }

    #[test]
    fn shared_globals_are_visible_through_rc() {
        let shared = globals(1000);
        let mut metrics = StyleMetrics::new(&STYLE_CLASSES[Style::LatnDflt.index()], shared.clone());
        metrics.scaler_mut().y_scale = 3 << 16;
        assert_eq!(shared.increase_x_height(), PROP_INCREASE_X_HEIGHT_MAX);
        assert_eq!(metrics.root().scaler.y_scale, 3 << 16);
        assert_eq!(Rc::strong_count(&shared), 2);
    }
}
