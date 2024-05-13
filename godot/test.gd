extends Node

var foo = Mat.from_array(2, 2, [1.0, 0.0, 0.0, 1.0])

var bar = Mat.identity(2, 2, "F64")

# Called when the node enters the scene tree for the first time.
func _ready():
	print(foo.add(bar))
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
