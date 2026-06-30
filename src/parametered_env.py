# Generic imports
import os
import numpy

from tensorforce.agents    import PPOAgent
from tensorforce.execution import Runner

# Custom imports
from environment import *

# Parameters not going into the environment
learning_frequency      = 50
batch_size              = learning_frequency
learning_rate           = 1.0e-3
gae_lambda              = 0.97
clipping_ratio          = 0.2
entropy                 = 0.01
model_dir               = '.'


def write_tandem_cylinder_reset(reset_dir, tandem_distance, radius=1.0, n_sampling_pts=10):
    if (tandem_distance <= 2.0*radius):
        print('Error : tandem_distance must be larger than two cylinder radii')
        exit()

    if (not os.path.exists(reset_dir)):
        os.makedirs(reset_dir)

    centers = [(-0.5*tandem_distance, 0.0),
               ( 0.5*tandem_distance, 0.0)]

    for body_id, center in enumerate(centers):
        cx, cy = center
        control_pts = [(cx+radius, cy),
                       (cx,        cy+radius),
                       (cx-radius, cy),
                       (cx,        cy-radius)]
        filename = reset_dir+'/body'+str(body_id)+'.csv'

        with open(filename, 'w') as file:
            file.write('4 {}\n'.format(n_sampling_pts))
            for i in range(0,4):
                file.write('0.5\n')
            for i in range(0,4):
                file.write('0.5\n')
            for pt in control_pts:
                file.write('{} {}\n'.format(pt[0], pt[1]))


# Define environment
def resume_env():
    # Environment parameters
    reset_dir               = 'reset/tandem_2cyl'
    geometry_mode           = 'tandem'
    baseline_geometry       = 'from_csv'  # Use 'from_csv' to preserve body0.csv/body1.csv
    nb_bodies               = 2
    nb_pts_to_move_per_body = 4
    pts_to_move_per_body    = [[0, 1, 2, 3],
                               [0, 1, 2, 3]]
    tandem_distance         = 4.0

    if (baseline_geometry == 'cylinder'):
        write_tandem_cylinder_reset(reset_dir, tandem_distance)
    elif (baseline_geometry == 'from_csv'):
        pass
    else:
        print('Unknown baseline geometry: '+baseline_geometry)
        exit()

    nb_pts_to_move          = nb_bodies * nb_pts_to_move_per_body
    pts_to_move             = pts_to_move_per_body
    nb_ctrls_per_episode    = 30
    nb_episodes             = 10000
    max_deformation         = 3.0
    restart_from_cylinder   = True
    replace_shape           = True
    comp_dir                = '.'
    restore_model           = False
    saving_model_period     = 10
    cfl                     = 0.5
    reynolds                = 100.0
    output_vtu              = True
    shape_h                 = 1.0
    domain_h                = 0.8
    cell_limit              = 50000
    xmin                    =-15.0
    xmax                    = 30.0
    ymin                    =-15.0
    ymax                    = 15.0
    final_time              = 2.0*(xmax-xmin)

    # Define environment
    environment=env(nb_pts_to_move, pts_to_move,
                    nb_ctrls_per_episode, nb_episodes,
                    max_deformation,
                    restart_from_cylinder,
                    replace_shape,
                    comp_dir,
                    restore_model,
                    saving_model_period,
                    final_time, cfl, reynolds,
                    output_vtu,
                    shape_h, domain_h,
                    cell_limit,
                    reset_dir,
                    xmin, xmax, ymin, ymax)

    return(environment)
