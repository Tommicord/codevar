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

//! Error type shared by every fallible widget operation.

use core::fmt;

use codevar_gpu_core::gpu_base::CompositorError;

use crate::texture::TextureId;
use crate::widget::{WidgetId, WidgetPhase};

/// Everything a widget, the tree or the renderer can report.
///
/// The tree and the control layer return this type from every fallible
/// call; Vulkan failures surface as [`CompositorError`] wrapped in
/// [`WidgetError::Compositor`] so the compositor's own error taxonomy
/// is preserved.
#[derive(Debug)]
pub enum WidgetError {
    /// An operation was attempted in a lifecycle state that does not
    /// allow it.
    InvalidState {
        /// The operation that was attempted (for example `mount`).
        operation: &'static str,
        /// The phase the widget tree was in.
        state: WidgetPhase,
    },
    /// A lifecycle transition was requested that the state machine
    /// does not define.
    InvalidTransition {
        /// Phase the transition started from.
        from: WidgetPhase,
        /// Phase the transition asked for.
        to: WidgetPhase,
    },
    /// Constraints were contradictory or not finite.
    InvalidConstraints,
    /// A geometry value was NaN or infinite where a finite value is
    /// required.
    NonFiniteValue,
    /// A child index was outside the child list.
    IndexOutOfBounds {
        /// Index that was addressed.
        index: usize,
        /// Number of children the widget has.
        len: usize,
    },
    /// A texture id is not registered.
    UnknownTexture {
        /// The unknown texture.
        id: TextureId,
    },
    /// Texture pixels do not match the advertised extent.
    InvalidTextureData {
        /// Byte length the format requires.
        expected: usize,
        /// Byte length that was supplied.
        actual: usize,
    },
    /// A hard limit of the widget system was exceeded.
    LimitExceeded {
        /// What limit was hit (for example `children`).
        what: &'static str,
        /// The limit that was hit.
        limit: usize,
    },
    /// The requested capability is not available in this build.
    Unsupported {
        /// What was requested.
        what: &'static str,
    },
    /// Shared widget state was already borrowed elsewhere; the caller
    /// may retry later.
    ControlBorrow,
    /// A widget referenced a widget that is not in the tree.
    UnknownWidget {
        /// The id that could not be resolved.
        id: WidgetId,
    },
    /// Vulkan or compositor failure while building or driving the
    /// render pipe.
    Compositor(CompositorError),
}

impl fmt::Display for WidgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState { operation, state } => {
                write!(f, "{operation} is not allowed while the tree is {state}")
            }
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid lifecycle transition from {from} to {to}")
            }
            Self::InvalidConstraints => f.write_str("layout constraints are contradictory"),
            Self::NonFiniteValue => f.write_str("a geometry value is not finite"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "child index {index} is out of bounds for {len} children")
            }
            Self::UnknownTexture { id } => write!(f, "texture {id} is not registered"),
            Self::InvalidTextureData { expected, actual } => {
                write!(f, "texture data has {actual} bytes but the format requires {expected}")
            }
            Self::LimitExceeded { what, limit } => {
                write!(f, "the {what} limit of {limit} was exceeded")
            }
            Self::Unsupported { what } => write!(f, "{what} is not supported"),
            Self::ControlBorrow => f.write_str("the widget state is already borrowed"),
            Self::UnknownWidget { id } => write!(f, "widget {id} is not part of the tree"),
            Self::Compositor(error) => write!(f, "compositor failure: {error}"),
        }
    }
}

impl core::error::Error for WidgetError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Compositor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CompositorError> for WidgetError {
    fn from(error: CompositorError) -> Self {
        Self::Compositor(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant renders a message that names the failing value,
    /// so logs stay actionable.
    #[test]
    fn display_names_the_failure() {
        let cases = [
            WidgetError::InvalidState {
                operation: "mount",
                state: WidgetPhase::Started,
            }
            .to_string(),
            WidgetError::InvalidTransition {
                from: WidgetPhase::Created,
                to: WidgetPhase::Disposed,
            }
            .to_string(),
            WidgetError::InvalidConstraints.to_string(),
            WidgetError::NonFiniteValue.to_string(),
            WidgetError::IndexOutOfBounds { index: 3, len: 2 }.to_string(),
            WidgetError::UnknownTexture {
                id: TextureId::new(7),
            }
            .to_string(),
            WidgetError::InvalidTextureData {
                expected: 4,
                actual: 3,
            }
            .to_string(),
            WidgetError::LimitExceeded {
                what: "children",
                limit: 64,
            }
            .to_string(),
            WidgetError::Unsupported { what: "shadows" }.to_string(),
            WidgetError::ControlBorrow.to_string(),
            WidgetError::UnknownWidget {
                id: WidgetId::new(1),
            }
            .to_string(),
            WidgetError::Compositor(CompositorError::Internal("test")).to_string(),
        ];
        for message in cases {
            assert!(!message.is_empty());
        }
        assert!(cases[0].contains("mount"));
        assert!(cases[3].contains("finite"));
        assert!(cases[5].contains('7'));
    }

    /// The compositor source is reachable through `Error::source`.
    #[test]
    fn compositor_source_is_exposed() {
        let error = WidgetError::Compositor(CompositorError::Internal("boom"));
        let source = core::error::Error::source(&error);
        assert!(source.is_some());
        assert!(WidgetError::ControlBorrow.source().is_none());
    }
}
