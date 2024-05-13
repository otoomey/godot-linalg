use godot::prelude::*;

mod mdata;
mod mat;

struct NumGDExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NumGDExtension {}
