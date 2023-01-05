SlicerAutoscoperM
-----------------

SlicerAutoscoperM is an extension for 3D Slicer for 2D-3D image registration integrating
with [Autoscoper][] for image-based 3D motion tracking of skeletal structures.

[Autoscoper]: https://github.com/BrownBiomechanics/Autoscoper

## Modules

| Name | Description |
|------|-------------|
| [AutoscoperM](AutoscoperM) | Allows to communicate with the Autoscoper process over a TCP connecion. |
| `AutoscoperMockServer` | Standalone executable implementing a mock Autoscoper TCP server. |

_:warning: This extension is in early development statge. Its content, API and behavior may change at any time. We mean it!_

## Python Linting

This project uses pre-commit and black for linting.
Install pre-commit with `pip install pre-commit` and setup with `pre-commit install`.
Linting will occur automatically when committing, or can be done explicitly with `pre-commit run --all-files`.

## Resources

To learn more about SlicerAutoscoperM, and Slicer, check out the following resources.

 - https://autoscoperm.slicer.org/
 - https://slicer.readthedocs.io/en/latest/


## Acknowledgments

See https://autoscoperm.slicer.org/acknowledgments


## License

This software is licensed under the terms of the [MIT](LICENSE).
