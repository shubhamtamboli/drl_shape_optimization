# Generic imports
import numpy as np

# Custom imports
from shapes_utils import *
from meshes_utils import *


def preserve_body_centers(shape_system, reference_centers):
    """Translate each body so that its center matches the reference center."""
    for body, ref_center in zip(shape_system.bodies, reference_centers):
        current_center = np.mean(body.control_pts, axis=0)
        body.control_pts += (ref_center - current_center)


### ************************************************
### Main execution
### Generate and mesh the tandem-cylinder baseline geometry.
reset_dir = 'reset/tandem_2cyl'
plot_pts  = True

shape = ShapeSystem()
shape.read_csv(reset_dir)

# Define the center-to-center distance between the imported bodies.
central_distance = 3
if len(shape.bodies) >= 2:
    centers = [np.mean(body.control_pts, axis=0) for body in shape.bodies[:2]]
    central_distance = float(np.linalg.norm(centers[0] - centers[1]))

    # Check for overlap using the maximum distance from each body center to its control points.
    body_radii = []
    for body in shape.bodies[:2]:
        body_center = np.mean(body.control_pts, axis=0)
        radius = np.max(np.linalg.norm(body.control_pts - body_center, axis=1))
        body_radii.append(float(radius))

    if central_distance <= sum(body_radii):
        print('Warning: the imported bodies intersect or overlap.')
else:
    print('Warning: expected at least two bodies to compute central distance.')

# Preserve the original body centers from the baseline geometry.
reference_centers = [np.mean(body.control_pts, axis=0) for body in shape.bodies]
preserve_body_centers(shape, reference_centers)

shape.generate(centering=False)
preserve_body_centers(shape, reference_centers)

shape.mesh(mesh_domain = True,
           shape_h     = 1.0,
           domain_h    = 1.0,
           xmin        =-15.0,
           xmax        = 30.0,
           ymin        =-15.0,
           ymax        = 15.0,
           mesh_format = 'mesh')
shape.generate_image(plot_pts=plot_pts,
                     xmin=-15.0,
                     xmax=30.0,
                     ymin=-15.0,
                     ymax=15.0)
shape.write_csv()