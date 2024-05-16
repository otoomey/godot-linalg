use godot::prelude::*;

mod mdata;
mod mat;
mod view;

struct NumGDExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NumGDExtension {}
