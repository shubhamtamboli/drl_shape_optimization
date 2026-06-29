# Generic imports
import os
import glob
import math
import time
import shutil
import numpy             as np

# Custom imports
from shapes_utils  import *
from meshes_utils  import *
from fenics_solver import *

# Define environment class for rl
class env():

    # Static variable
    episode_nb =-1
    control_nb = 0

    # Initialize empty shape system
    shape = ShapeSystem()

    def __init__(self,
                 nb_pts_to_move, pts_to_move,
                 nb_ctrls_per_episode, nb_episodes,
                 max_deformation,
                 restart_from_cylinder,
                 replace_shape,
                 comp_dir,
                 restore_model,
                 saving_model_period,
                 final_time, cfl, reynolds,
                 output,
                 shape_h, domain_h,
                 cell_limit,
                 reset_dir,
                 xmin, xmax, ymin, ymax):

        self.pts_to_move             = self.normalize_pts_to_move(pts_to_move)
        self.nb_bodies               = len(self.pts_to_move)
        self.nb_pts_to_move          = sum(len(pts) for pts in self.pts_to_move)
        self.nb_ctrls_per_episode    = nb_ctrls_per_episode
        self.nb_episodes             = nb_episodes
        self.max_deformation         = max_deformation
        self.restart_from_cylinder   = restart_from_cylinder
        self.replace_shape           = replace_shape
        self.comp_dir                = comp_dir
        self.restore_model           = restore_model
        self.final_time              = final_time
        self.cfl                     = cfl
        self.reynolds                = reynolds
        self.output                  = output
        self.shape_h                 = shape_h
        self.domain_h                = domain_h
        self.cell_limit              = cell_limit
        self.reset_dir               = reset_dir
        self.xmin                    = xmin
        self.xmax                    = xmax
        self.ymin                    = ymin
        self.ymax                    = ymax

        # Saving model periodically
        env.saving_model_period = saving_model_period

        # Check that reset dir exists
        if (not os.path.exists('./'+self.reset_dir)):
            print('Error : I could not find the reset folder')
            exit()

        # Initialize shape system by reading it from reset folder
        env.shape = ShapeSystem()
        env.shape.read_csv(self.reset_dir)
        env.shape.generate(centering=False)

        # Initialize arrays
        self.drag       = np.array([])
        self.lift       = np.array([])
        self.reward     = np.array([])
        self.avg_drag   = np.array([])
        self.avg_lift   = np.array([])
        self.avg_reward = np.array([])
        self.penal      = np.array([])

        # If restore model, get last increment
        if (self.restore_model):
            file_lst        = glob.glob(self.comp_dir+'/save/png/*.png')
            last_file       = max(file_lst, key=os.path.getctime)
            tmp             = last_file.split('_')[-1]
            env.shape.index = int(tmp.split('.')[0])
            print('Restarting from shape index '+str(env.shape.index))

        # Remove save folder
        if (not self.restore_model):
            save_dir = self.comp_dir+'/save'
            if (os.path.exists(save_dir)):
                shutil.rmtree(save_dir)

            # Make sure the save repo exists and is properly formated
            os.makedirs(save_dir+'/png',      exist_ok=True)
            os.makedirs(save_dir+'/rejected', exist_ok=True)
            os.makedirs(save_dir+'/xml',      exist_ok=True)
            os.makedirs(save_dir+'/csv',      exist_ok=True)
            os.makedirs(save_dir+'/sol',      exist_ok=True)

            # Copy initial files in save repo if restart from cylinder/tandem baseline
            if (self.restart_from_cylinder):
                for filename in glob.glob(self.reset_dir+'/*.png'):
                    shutil.copy(filename, save_dir+'/png/.')
                for filename in glob.glob(self.reset_dir+'/*.xml'):
                    shutil.copy(filename, save_dir+'/xml/.')
                for filename in glob.glob(self.reset_dir+'/*.csv'):
                    shutil.copy(filename, save_dir+'/csv/.')

    def normalize_pts_to_move(self, pts_to_move):
        if (len(pts_to_move) == 0):
            return []
        if isinstance(pts_to_move[0], (list, tuple, np.ndarray)):
            return [list(pts) for pts in pts_to_move]
        return [list(pts_to_move)]

    def reset(self):
        # Console output
        env.episode_nb += 1
        print('****** Starting episode '+str(env.episode_nb))
        if (env.episode_nb%100 == 0): time.sleep(10)

        # Reset control number
        env.control_nb  = 0

        # Reset from baseline if asked
        if (self.restart_from_cylinder):
            env.shape.read_csv(self.reset_dir, keep_numbering=True)
            env.shape.generate(centering=False)

        # Fill next state
        next_state = self.fill_next_state(True, 0)

        return(next_state)

    def execute(self, action=None):
        # Console output
        print('***    Starting control '+str(env.control_nb))

        # Convert actions to body-wise numpy arrays
        action = np.array(action)
        deformation = action.reshape((int(len(action)/3), 3))
        body_deformation = []
        action_index = 0

        for body_id, pts in enumerate(self.pts_to_move):
            body = env.shape.bodies[body_id]
            body_center = np.mean(body.control_pts, axis=0)
            current_body_deformation = np.zeros((len(pts), 3))

            for local_id, pt in enumerate(pts):
                radius = max(abs(deformation[action_index,0]),0.2)*self.max_deformation
                dangle = (360.0/float(body.n_control_pts))
                angle  = dangle*float(pt)+deformation[action_index,1]*dangle/2.0
                x      = body_center[0] + radius*math.cos(math.radians(angle))
                y      = body_center[1] + radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation[action_index,2])

                current_body_deformation[local_id,0] = x
                current_body_deformation[local_id,1] = y
                current_body_deformation[local_id,2] = edg
                action_index += 1

            body_deformation.append(current_body_deformation)

        # Modify shape system
        env.shape.modify_shape_from_field(body_deformation,
                                          replace=self.replace_shape,
                                          pts_list=self.pts_to_move)
        env.shape.generate(centering=False)
        env.shape.write_csv()

        try:
            meshed, n_tri = env.shape.mesh(mesh_domain = True,
                                           shape_h     = self.shape_h,
                                           domain_h    = self.domain_h,
                                           xmin        = self.xmin,
                                           xmax        = self.xmax,
                                           ymin        = self.ymin,
                                           ymax        = self.ymax,
                                           mesh_format = 'xml')

            # Do not solve if mesh is too large
            if (n_tri > self.cell_limit):
                meshed = False
        except Exception as exc:
            print(exc)
            meshed = False

        # Generate image
        env.shape.generate_image(plot_pts    = True,
                                 quad_radius = self.max_deformation,
                                 xmin        = self.xmin,
                                 xmax        = self.xmax,
                                 ymin        = self.ymin,
                                 ymax        = self.ymax)

        png_name = env.shape.name+'_'+str(env.shape.index)+'.png'
        if (not meshed):
            shutil.copy(png_name, self.comp_dir+'/save/rejected/.')

        # Save png and csv files
        shutil.move(png_name, self.comp_dir+'/save/png/.')
        for filename in glob.glob(env.shape.name+'_'+str(env.shape.index)+'_body*.csv'):
            shutil.move(filename, self.comp_dir+'/save/csv/.')

        # Copy new shape files to save folder
        xml_name = env.shape.name+'_'+str(env.shape.index)+'.xml'
        if (meshed):
            shutil.copy(xml_name, self.comp_dir+'/save/xml/.')

        # Update control number
        env.control_nb += 1

        # Compute reward with try/catch
        self.compute_reward(meshed)

        # Save quantities of interest
        self.save_qoi()

        # Fill next state
        next_state = self.fill_next_state(meshed, env.shape.index)

        # Copy u, v and p solutions to repo if the solver produced them
        if (meshed and self.last_solved):
            for suffix in ['_u.png', '_v.png', '_p.png']:
                filename = str(env.shape.index)+suffix
                if os.path.exists(filename):
                    shutil.move(filename, self.comp_dir+'/save/sol/.')

        # Remove mesh file from repo
        if (meshed):
            os.remove(xml_name)

        # Return
        terminal = False
        print("good epoch; reward: {}".format(self.reward[-1]))
        return(next_state, terminal, self.reward[-1])

    def compute_reward(self, meshed):
        self.last_solved = False

        # If meshing was successful, reward is computed normally
        if (meshed):
            try:
                # Compute drag and lift
                name = self.comp_dir+'/'+env.shape.name+'_'+str(env.shape.index)+'.xml'
                pts  = env.shape.get_control_pts()
                drag, lift, solved = solve_flow(mesh_file      = name,
                                                final_time     = self.final_time,
                                                reynolds       = self.reynolds,
                                                output         = self.output,
                                                cfl            = self.cfl,
                                                pts_x          = pts[:,0],
                                                pts_y          = pts[:,1],
                                                obstacle_boxes = env.shape.get_obstacle_boxes(),
                                                xmin           = self.xmin,
                                                xmax           = self.xmax,
                                                ymin           = self.ymin,
                                                ymax           = self.ymax)
                # Save solution png if the solver produced the summary image
                summary_png = str(env.shape.index)+'.png'
                if os.path.exists(summary_png):
                    shutil.move(summary_png, self.comp_dir+'/save/sol/.')
            except Exception as exc:
                print(exc)
                solved = False

            # If solver was successful
            if (solved):
                self.last_solved = True

                # Drag is always <0 while lift changes sign
                penal  = 0.0
                lift   =-lift # Make lift positive
                if (lift > 2.0): lift=2.0*lift # Shaping for faster convergence
                reward = lift/abs(drag)
                reward = max(reward, -10.0)

            # If solver was not successful
            else:
                drag   =-1.0
                lift   = 0.0
                reward =-5.0
                penal  = 5.0

        # If meshing was not successful, we just return a high penalization
        else:
            drag   =-1.0
            lift   = 0.0
            reward =-5.0
            penal  = 5.0

        # Save drag, lift, reward and penalization
        self.drag   = np.append(self.drag,   drag)
        self.lift   = np.append(self.lift,   lift)
        self.reward = np.append(self.reward, reward)
        self.penal  = np.append(self.penal,  penal)

        denom      = max(env.shape.index, 1)
        val_drag   = np.sum(self.drag)/denom
        val_lift   = np.sum(self.lift)/denom
        val_reward = np.sum(self.reward)/denom
        self.avg_drag   = np.append(self.avg_drag,   val_drag)
        self.avg_lift   = np.append(self.avg_lift,   val_lift)
        self.avg_reward = np.append(self.avg_reward, val_reward)

    def save_qoi(self):
        # Retrieve current index
        i = env.shape.index

        # Write drag/lift values to file
        filename = self.comp_dir+'/save/drag_lift'
        with open(filename, 'a') as f:
            f.write('{} {} {} {} {}\n'.format(i,
                                              self.drag[-1],
                                              self.lift[-1],
                                              self.avg_drag[-1],
                                              self.avg_lift[-1]))

        # Write reward and penalization to file
        filename = self.comp_dir+'/save/reward_penalization'
        with open(filename, 'a') as f:
            f.write('{} {} {} {}\n'.format(i,
                                           self.reward[-1],
                                           self.penal[-1],
                                           self.avg_reward[-1]))

    def fill_next_state(self, meshed, index):
        return env.shape.get_state()

    @property
    def states(self):
        return dict(
            type='float',
            shape=(len(env.shape.get_state())))

    @property
    def actions(self):
        return dict(
            type='float',
            shape=(self.nb_pts_to_move*3),
            min_value=-1.0,
            max_value= 1.0)