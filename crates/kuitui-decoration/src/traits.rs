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

//! The traits a decoration implements and the traits a decoration host
//! provides.
//!
//! # The contract
//!
//! A decoration implements [`Decoration`]. Every hook has a default
//! implementation, so a decoration only overrides what it needs: a
//! decoration that only paints a border implements [`Decoration::draw`]
//! and nothing else.
//!
//! A host provides [`DecorationBackend`]: the native windowing system
//! calls the host to create surfaces, set titles, change window styles
//! and deliver pointer events. A host that only paints and never touches
//! the native API implements [`DecorationBackend`] with every method
//! returning [`DecorationError::Unavailable`].

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;

use codevar_kuitui::widget::Widget;

use crate::error::{DecorationError, DecorationResult};
use crate::modifiers::{DecorationModifier, DecorationModifiers};
use crate::state::DecorationState;
use crate::theme::DecorationTheme;
use crate::types::{
    DecorationBounds, DecorationIcon, DecorationKind, DecorationMetrics, DecorationResizeEdge,
};

/// The native surface a decoration paints into.
///
/// A backend creates one surface per decorated window. A surface is
/// identified by its id and carries the bounds, the modifiers and the
/// state the windowing system assigned to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecorationSurface {
    /// The id of the surface.
    pub id: crate::types::DecorationId,
    /// The bounds the windowing system assigned to the surface.
    pub bounds: DecorationBounds,
    /// The modifiers that are active on the surface.
    pub modifiers: DecorationModifiers,
}

impl DecorationSurface {
    /// Creates a surface with no bounds and no modifiers.
    #[must_use]
    pub const fn new(id: crate::types::DecorationId) -> Self {
        Self {
            id,
            bounds: crate::types::DecorationBounds::EMPTY,
            modifiers: DecorationModifiers::NONE,
        }
    }
}

/// A widget that decorates a window.
///
/// A decoration is both a [`codevar_kuitui::widget::Widget`] and a
/// native surface: it measures, places, draws and handles events like
/// any other widget, and it also reports which decoration modifiers it
/// supports and which are active.
pub trait Decoration: Widget {
    /// The kind of the decoration.
    #[must_use]
    fn kind(&self) -> DecorationKind;

    /// The id of the decoration, or `None` when it is not bound to a
    /// native surface yet.
    #[must_use]
    fn id(&self) -> Option<crate::types::DecorationId>;

    /// Binds the decoration to a native surface.
    fn bind(&mut self, id: crate::types::DecorationId);

    /// Unbinds the decoration from its native surface.
    fn unbind(&mut self);

    /// The modifiers this decoration supports changing.
    ///
    /// The default is every modifier the type knows about.
    #[must_use]
    fn supported_modifiers(&self) -> DecorationModifiers {
        DecorationModifiers::ALL
    }

    /// The modifiers that are currently active on the decoration.
    ///
    /// The default reports none.
    #[must_use]
    fn active_modifiers(&self) -> DecorationModifiers {
        DecorationModifiers::NONE
    }

    /// Whether `modifier` is active on the decoration.
    #[must_use]
    fn is_active(&self, modifier: DecorationModifier) -> bool {
        self.active_modifiers().has(modifier)
    }

    /// The state of the decoration.
    #[must_use]
    fn state(&self) -> DecorationState {
        DecorationState {
            active: self.active_modifiers(),
            supported: self.supported_modifiers(),
        }
    }

    /// Sets the active modifiers of the decoration.
    ///
    /// A backend calls this when the windowing system reports a state
    /// change. The default stores nothing and returns `Ok`.
    fn set_active_modifiers(&mut self, _modifiers: DecorationModifiers) -> DecorationResult<()> {
        Ok(())
    }

    /// Requests that `modifier` be changed on the native surface.
    ///
    /// A backend calls this when the user toggled a modifier through a
    /// decoration widget (for example the maximize button of a title
    /// bar). The default returns [`DecorationError::Unavailable`].
    fn request_modifier(&mut self, _modifier: DecorationModifier, _enable: bool) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// The bounds of the decoration.
    #[must_use]
    fn bounds(&self) -> DecorationBounds {
        DecorationBounds::default()
    }

    /// Sets the bounds of the decoration.
    fn set_bounds(&mut self, _bounds: DecorationBounds) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// The metrics of the decoration.
    #[must_use]
    fn metrics(&self) -> DecorationMetrics {
        DecorationMetrics::default()
    }

    /// Sets the metrics of the decoration.
    fn set_metrics(&mut self, _metrics: DecorationMetrics) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// The icon of the decoration.
    #[must_use]
    fn icon(&self) -> Option<&DecorationIcon> {
        None
    }

    /// Sets the icon of the decoration.
    fn set_icon(&mut self, _icon: DecorationIcon) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// The resize edge the decoration covers, or `None` when it is not a
    /// resize edge.
    #[must_use]
    fn resize_edge(&self) -> Option<DecorationResizeEdge> {
        None
    }

    /// Whether the decoration draws a border around its bounds.
    #[must_use]
    fn draws_border(&self) -> bool {
        false
    }

    /// The color of the border, when [`Self::draws_border`] returns
    /// `true`.
    #[must_use]
    fn border_color(&self, _theme: &DecorationTheme) -> codevar_kuitui::color::Color {
        codevar_kuitui::color::Color::TRANSPARENT
    }
}

/// The backend a decoration host provides.
///
/// A host owns the native windowing system: it creates surfaces, sets
/// titles and icons, changes window styles and delivers pointer events.
/// Every method has a default implementation that returns
/// [`DecorationError::Unavailable`], so a host that only paints does not
/// have to implement anything.
pub trait DecorationBackend {
    /// Creates a native surface for `decoration` and returns its id.
    ///
    /// The default returns [`DecorationError::Unavailable`].
    fn create_surface(
        &mut self,
        _decoration: &dyn Decoration,
    ) -> DecorationResult<crate::types::DecorationId> {
        Err(DecorationError::Unavailable)
    }

    /// Destroys the native surface behind `id`.
    ///
    /// The default returns [`DecorationError::Unavailable`].
    fn destroy_surface(&mut self, _id: crate::types::DecorationId) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// Sets the title of the surface behind `id`.
    ///
    /// The default returns [`DecorationError::Unavailable`].
    fn set_title(&mut self, _id: crate::types::DecorationId, _title: &str) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// Sets the icon of the surface behind `id`.
    ///
    /// The default returns [`DecorationError::Unavailable`].
    fn set_icon(&mut self, _id: crate::types::DecorationId, _icon: &DecorationIcon) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// Sets the bounds of the surface behind `id`.
    ///
    /// The default returns [`DecorationError::Unavailable`].
    fn set_bounds(
        &mut self,
        _id: crate::types::DecorationId,
        _bounds: DecorationBounds,
    ) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// Changes the modifiers of the surface behind `id`.
    ///
    /// The default returns [`DecorationError::Unavailable`].
    fn set_modifiers(
        &mut self,
        _id: crate::types::DecorationId,
        _modifiers: DecorationModifiers,
    ) -> DecorationResult<()> {
        Err(DecorationError::Unavailable)
    }

    /// Reports the metrics the windowing system assigned to `id`.
    ///
    /// The default returns [`DecorationMetrics::default`].
    fn metrics(&self, _id: crate::types::DecorationId) -> DecorationMetrics {
        DecorationMetrics::default()
    }

    /// Delivers a pointer event to the surface behind `id`.
    ///
    /// The default does nothing.
    fn pointer_event(
        &mut self,
        _id: crate::types::DecorationId,
        _event: &codevar_kuitui::event::PointerEvent,
    ) {
    }

    /// Delivers a key event to the surface behind `id`.
    ///
    /// The default does nothing.
    fn key_event(&mut self, _id: crate::types::DecorationId, _event: &codevar_kuitui::event::KeyEvent) {}
}

/// A host of decorations: it owns the backend and the decorations it
/// created.
pub struct DecorationHost<B: DecorationBackend> {
    /// The native backend the host talks to.
    pub backend: B,
    /// The decorations the host owns, keyed by their id.
    pub decorations: Vec<(crate::types::DecorationId, Rc<dyn Decoration>)>,
}

impl<B: DecorationBackend> DecorationHost<B> {
    /// Creates a host that talks to `backend`.
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            decorations: Vec::new(),
        }
    }

    /// Registers a decoration and binds it to a native surface.
    ///
    /// Returns the id of the decoration.
    pub fn register(
        &mut self,
        decoration: Box<dyn Decoration>,
    ) -> DecorationResult<crate::types::DecorationId> {
        let id = crate::types::DecorationId::new(self.decorations.len() as u64 + 1);
        self.backend.create_surface(decoration.as_ref())?;
        let mut decoration = decoration;
        decoration.bind(id);
        let decoration = Rc::from(decoration);
        self.decorations.push((id, decoration));
        Ok(id)
    }

    /// Unregisters the decoration with `id` and destroys its surface.
    pub fn unregister(&mut self, id: crate::types::DecorationId) -> DecorationResult<()> {
        if let Some(index) = self
            .decorations
            .iter()
            .position(|(decoration_id, _)| *decoration_id == id)
        {
            let (_, mut decoration) = self.decorations.swap_remove(index);
            if let Some(decoration) = Rc::get_mut(&mut decoration) {
                decoration.unbind();
            }
            self.backend.destroy_surface(id)?;
        }
        Ok(())
    }

    /// Returns the decoration with `id`, if any.
    #[must_use]
    pub fn decoration(&self, id: crate::types::DecorationId) -> Option<&Rc<dyn Decoration>> {
        self.decorations
            .iter()
            .find(|(decoration_id, _)| *decoration_id == id)
            .map(|(_, decoration)| decoration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DecorationKind;
    use codevar_kuitui::geometry::{Constraints, Size};
    use codevar_kuitui::widget::MeasureScope;

    struct NoopBackend;

    impl DecorationBackend for NoopBackend {
        fn create_surface(
            &mut self,
            _decoration: &dyn Decoration,
        ) -> DecorationResult<crate::types::DecorationId> {
            Ok(crate::types::DecorationId::new(1))
        }

        fn destroy_surface(&mut self, _id: crate::types::DecorationId) -> DecorationResult<()> {
            Ok(())
        }
    }

    #[test]
    fn a_host_creates_and_destroys_surfaces_through_its_backend() {
        let mut host = DecorationHost::new(NoopBackend);
        let decoration = Box::new(NullDecoration);
        let id = host.register(decoration).unwrap();
        assert!(host.decoration(id).is_some());
        host.unregister(id).unwrap();
        assert!(host.decoration(id).is_none());
    }

    struct NullDecoration;

    impl Widget for NullDecoration {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::ZERO)
        }
    }

    impl Decoration for NullDecoration {
        fn kind(&self) -> DecorationKind {
            DecorationKind::TitleBar
        }

        fn id(&self) -> Option<crate::types::DecorationId> {
            None
        }

        fn bind(&mut self, _id: crate::types::DecorationId) {}

        fn unbind(&mut self) {}
    }
}
