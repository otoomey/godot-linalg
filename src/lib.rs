use godot::prelude::*;

mod matrix;

struct NumGDExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NumGDExtension {}
