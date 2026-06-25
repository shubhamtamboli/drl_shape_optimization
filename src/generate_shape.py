# Custom imports
from shapes_utils import *
from meshes_utils import *

### ************************************************
### Main execution
### Generate and mesh the tandem-cylinder baseline geometry.
reset_dir = 'reset/tandem_2cyl'
plot_pts  = True

shape = ShapeSystem()
shape.read_csv(reset_dir)
shape.generate(centering=False)
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