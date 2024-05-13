_This extension is in early development_

# Linear algebra library for Godot
This extension wraps Rust's [nalgebra](https://nalgebra.org/) crate to make it usuable in GDScript.

## Motivation
GDScript, in its current state, is a lot of the things that are not great about Python with very little of the things that we like about Python. This extension adds two features to GDScript:
* Arbitrary matrix and vector sizes.
* New matrix datatypes, such as `u8`, as well as complex types such as `c64` and `c128`.

## Example
```gdscript
var a = Mat.identity(2, 2, 'U8').astype('F32')
var b = Mat.from_array(2, 2, [0.0, 2.0, 0.0, 2.0]).astype('F32')

print(a + b)

# produces
#  ┌     ┐
#  │ 1 2 │
#  │ 2 1 │
#  └     ┘
```
