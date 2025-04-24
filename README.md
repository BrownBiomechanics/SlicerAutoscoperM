SlicerAutoscoperM
-----------------

SlicerAutoscoperM is an extension for 3D Slicer for 2D-3D image registration integrating
with [Autoscoper][] for image-based 3D motion tracking of skeletal structures.

[Autoscoper]: https://github.com/BrownBiomechanics/Autoscoper

## Modules

| Name | Description |
|------|-------------|
| [AutoscoperM](AutoscoperM) | This module integrates Autoscoper, an open source application for 3D tracking of skeletal structures in single-, bi- and multi-plane videoradiography. The AutoscoperM module also includes pre-processing functionalities for Autoscoper videoradiography inputs and for the 3D Hierarchical Registration module. |
| [Hierarchical3DRegistration](Hierarchical3DRegistration) | A module for registration of rigid skeletal and implant motion from static and dynamic computed tomography (3DCT and 4DCT) |
| [Tracking Evaluation](TrackingEvaluation) | A module for comparison of the Autoscoper tracking results against ground truth data. |

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
