import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import math
import time

# import MLD_hybrid_STL_Gurobi as MLD
# import MLD_hybrid_STL_Drake as MLD
# import Biped_Drake as MLD
# import Biped_MPC as MPC

import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加lib目录
lib_path = os.path.join(project_root, 'lib')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import DE_cuda_solver

# x_lim = [{'xl':   0, 'xu':   3.5, 'yl':   0, 'yu':  10, 'color': 'red'   }, #0
#          {'xl':   3.5, 'xu':   6.5, 'yl':   0, 'yu':   3.5, 'color': 'orange'}, #1
#          {'xl':   6.5, 'xu':  10, 'yl':   0, 'yu':  10, 'color': 'green' }, #2
#          {'xl':   3.5, 'xu':   6.5, 'yl':   6.5, 'yu':   10, 'color': 'yellow'}, #3
#          {'xl':   0, 'xu':   1, 'yl':   4.5, 'yu':   5.5, 'color': 'blue'  }, #4
#          {'xl':   4.5, 'xu':   5.5, 'yl':   9, 'yu':   10, 'color': 'black' }, #5
#          {'xl':   9, 'xu':   10, 'yl': 4.5, 'yu': 5.5, 'color': 'purple'}] #6

x_lim = [{'xl':   0, 'xu':   1., 'yl':   0, 'yu':  3, 'color': 'lightcoral'   }, #0
         {'xl':   1., 'xu':   2., 'yl':   0, 'yu':   1., 'color': 'orange'}, #1
         {'xl':   2, 'xu':  3, 'yl':   0, 'yu':  3, 'color': 'lime' }, #2
         {'xl':   1, 'xu':   2, 'yl':   2, 'yu':   3, 'color': 'yellow'}, #3
         {'xl':   0, 'xu':   0.2, 'yl':   1.4, 'yu':   1.6, 'color': 'lightcyan'  }, #4
         {'xl':   1.4, 'xu':   1.6, 'yl':   2.8, 'yu':   3, 'color': 'greenyellow' }, #5
         {'xl':   2.8, 'xu':   3, 'yl': 1.4, 'yu': 1.6, 'color': 'pink'}] #6


num_patch = len(x_lim)
# x0_MLD = np.array([0.45, 0.5, 0, 0.0, math.pi/2])
# x0_MLD = np.array([1.0*0.3, 1.0*0.3, 0, 0., math.pi/2])

# Roman's infeasible solutions
x0_MLD = np.array([ 0.29357406,  0.29125562, -0.01193462, -0.01774755,  1.58432257])

nu = 3
nx = 5
nz = num_patch

nc = 4*num_patch+6+6+4+2*3
N = 20

h = np.zeros([nc, 1])
h_d_theta = np.zeros([nx, 1])

Mx = 10; My = 10
Mu = 5
Mt = 7
M = [Mx, My, Mu, Mt]

i_con = 0

# Constraints on v limits
h[i_con:(i_con+4)] = np.array([[0.5],
                               [0.5],
                               [0.5],
                               [0.5]])

i_con += 4

# Constraints on theta limites
h[i_con:(i_con+2)] = np.array([[math.pi*5/4],
                               [math.pi*5/4]])
i_con += 2

# Constraints on control limits: p_foot_x, p_foot_y, theta_dot
h[i_con:(i_con+6)] = np.array([[0.25],
                               [0.25],
                               [0.25],
                               [0.25],
                               [math.pi/12],
                               [math.pi/12]])

i_con += 6

# Constraints on patches
for ii in range(num_patch):
    h[(i_con+ii*4+0):(i_con+ii*4+4)] = np.array([[ x_lim[ii]['xu']+Mx],
                                                 [-x_lim[ii]['xl']+Mx],
                                                 [ x_lim[ii]['yu']+My],
                                                 [-x_lim[ii]['yl']+My]])

i_con += num_patch*4

# Constraints on standing mode selection: Vx and Vy == 0
h[i_con:(i_con+4)] = np.array([[Mu],
                               [Mu],
                               [Mu],
                               [Mu]])
i_con += 4

# Constraints on orientation selection
# breakpoints = [-math.pi -math.pi/4+ (ii)/4 *math.pi for ii in range(nk+1)]   # breakpoints = [-math.pi, -math.pi * 2 / 3, -math.pi / 3, 0, math.pi / 3, math.pi * 2 / 3, math.pi]
for ii in range(3):
    h[(i_con+ii*2+0):(i_con+ii*2+2)] = np.array([[(math.pi - ii*math.pi/2) + Mt],
                                                 [-(math.pi - ii*math.pi/2) + Mt]])
i_con += 2*3

# for ii in range(nk):
#     h[(i_con+ii*2+0):(i_con+ii*2+2)] = np.array([[ breakpoints[ii+1]+Mt],
#                                                  [-breakpoints[ii]  +Mt]])
# i_con += 2*nk
# Specific direction of interest: -pi("8") | -pi/2("9") | 0("10") | pi/2("11") | pi("12")
# for ii in range(5):
#     h[(i_con+ii*2+0):(i_con+ii*2+2)] = np.array([[(-math.pi + ii*math.pi/2)+0.15  + Mt],
#                                                  [-(-math.pi + ii*math.pi/2) + Mt]])
# i_con += 2*5
# print("nc size: ", nc)
# print("h: ", h)

def plot_result(sol_x, sol_u, color, linestyle):
    ax = plt.gca()
    ax.set_aspect('equal') 
    ax.set_facecolor("gray")

    # Add patches
    for item in x_lim:
        rectangle = plt.Rectangle((item['xl'], item['yl']), (item['xu']-item['xl']), (item['yu']-item['yl']), fc=item['color'], zorder=2)
        ax.add_patch(rectangle)

    # Add CoM locations, color based on current footstep
    for i in range(0, sol_x.shape[0]):
        if i % 2 == 0:
            ax.scatter(sol_x[i, 0], sol_x[i, 1], s=3, zorder=3, color="grey")
        else: 
            ax.scatter(sol_x[i, 0], sol_x[i, 1], s=3, zorder=3, color="lavender")
    
    # Add heading angles as lines coming out of CoM points
    ang = sol_x[:,4]
    ax.plot([sol_x[:, 0], sol_x[:, 0]+0.15*np.cos(ang)], [sol_x[:, 1] , sol_x[:, 1]+0.15*np.sin(ang)], color=color, alpha=0.5, linewidth=0.3) # directions
    # ax.plot(sol_x[:-1, 0] + sol_u[:-1, 0], sol_x[:-1, 1] + sol_u[:-1, 1], linestyle='--', color='blue', alpha=0.8, linewidth=0.5) # dashed line connecting footstep
    ax.plot(sol_x[:-1, 0] + sol_u[:, 0], sol_x[:-1, 1] + sol_u[:, 1], linestyle='--', color='blue', alpha=0.8, linewidth=0.5) # dashed line connecting footstep
    
    # Add footstep dots
    for i in range(sol_u.shape[0]): 
        if i % 2 == 0:
            ax.scatter(sol_x[i, 0] + sol_u[i, 0], sol_x[i, 1] + sol_u[i, 1], marker='*', color="grey", s=3, zorder=3)
        else: 
            ax.scatter(sol_x[i, 0] + sol_u[i, 0], sol_x[i, 1] + sol_u[i, 1], marker='*', color="lavender", s=3, zorder=3)

    # Set CoM color to red if robot stops at that point
    for sol in sol_x:
        if sol[2] == 0 and sol[3] == 0:
            ax.scatter(sol[0], sol[1], color="red", s=3, zorder=3)
            
    # Add arrow to indicate vel. direction
    show_vel_arrows = True
    if show_vel_arrows:
        scale = 0.33
        for i in range(len(sol_x)):
            plt.arrow(sol_x[i, 0], sol_x[i, 1], sol_x[i, 2] * scale, sol_x[i, 3] * scale, 
                    head_width=0.01, head_length=0.02, fc='r', ec='r', zorder=3)

    # Add arrow to indicate vel. direction
    show_vel_arrows = True
    if show_vel_arrows:
        scale = 0.33
        for i in range(len(sol_x)):
            plt.arrow(sol_x[i, 0], sol_x[i, 1], sol_x[i, 2] * scale, sol_x[i, 3] * scale, 
                    head_width=0.01, head_length=0.02, fc='r', ec='r', zorder=3)

    # plt.show()

print("=============Start Planning=============")

solver = DE_cuda_solver.Create()
# print("Solver object created successfully")

solver.init_solver(0)
# print("Solver initialized successfully")

solve_start = time.time()

# solution = solver.Solve()

solver_time = time.time() - solve_start
print("solver time:", solver_time)

# MLD_solver = MLD.GBD(nx, nu, nz, nc, N, "factory", h, M, False)
# sol = MLD_solver.solve_full_problem(x0_MLD)
# sol_x = np.zeros([N+1, nx])
# for ii in range(N+1):
#     for ix in range(nx):
#         sol_x[ii, ix] = sol['x_' + str(ii) + '_' + str(ix)]
        
# sol_u = np.zeros([N, nu])
# for ii in range(N):
#     for iu in range(nu):
#         sol_u[ii, iu] = sol['u_' + str(ii) + '_' + str(iu)]    

# # sol_u = np.array([sol['control_x'], sol['control_y']])

fitness = solution["fitness"]
objective_score = solution["objective_score"]
constraint_score = solution["constraint_score"]
sol_x = solution["state"]
sol_u = solution["param"]

print("fitness:",fitness)
print("objective_score:",objective_score)
print("constraint_score:",constraint_score)

# sol_x = sol_x.reshape
print(len(sol_x))
print(len(sol_u))
sol_x = np.reshape(sol_x, (N, 5))
sol_x = np.vstack([x0_MLD, sol_x])
sol_u = np.reshape(sol_u, (N, 3))

print("solution x",sol_x)
print("solution u",sol_u)

# plot results
plot_result(sol_x, sol_u, 'black', '--')
plt.grid()
plt.savefig(f'figures/MLD_planning', dpi=400, bbox_inches="tight")
print("=============Finish Planning=============")

# print ("Skipping MPC")
# quit()

# print("Solving MPC")
# MPC_solver = MPC.GBD(nx, nu, nz, nc, N, "factory", h, M, False)
# print("here1")
# time_consumed = []

# dT_dyn = 0.2

# # real_traj = np.array([
# #     [1.0*0.4, 1.0*0.4, 0, 0.0, math.pi/2],
# #     [0.407218,0.427687,0.144986,0.266730,1.667635],
# #     [0.373711,0.467882,0.106846,-0.191450,1.684957],
# #     [0.451151,0.519353,0.164696,-0.067641,1.664548]])

# # real_traj = np.array([
# #     [0.37413114309310913, 0.315681517124176, -0.024738911539316177, 0.003569994820281863, 1.5475488901138306]])


# real_traj_x = np.array([[ 0.29357406,  0.29125562, -0.01193462, -0.01774755,  1.58432257],
#                         [ 0.20256279,  0.29650193, -0.55253625,  0.03899408,  1.4822861],
#                         [ 0.16901538,  0.3842341,  0.23552534,  0.48294446,  1.44902432],
#                         [ 0.1043901,   0.53509599, -0.54894269,  0.45836598,  1.40276766],
#                         [ 0.14855266,  0.66786045,  0.66206676,  0.325573,    1.49501479],
#                         [ 0.16124251,  0.80636841, -0.39470398,  0.44708541,  1.60090089],
#                         [ 0.17617033,  0.92104822,  0.66809273,  0.21478182,  1.80559075],
#                         [0, 0, 0, 0, 0]]) # Last array is filler so no error occurs when we draw

# real_traj_u = np.array([[ 0.11881212, -0.09006991,  0.        ],
#                         [-0.23456815, -0.09912089, -0.65541023],
#                         [ 0.2115667,   0.03448935,  1.14130962],
#                         [-0.3172552,   0.0523746,  -0.38620454],
#                         [ 0.2739979,   0.01103512,  0.14228131],
#                         [-0.25070429,  0.07685937, -0.24169903],
#                         [ 0.27423054, -0.0543599,   1.03166974]])

# print(f"real_traj_x shape: {real_traj_x.shape}, dtype: {real_traj_x.dtype}, flags: {real_traj_x.flags}")
# print(f"real_traj_u shape: {real_traj_u.shape}, dtype: {real_traj_u.dtype}, flags: {real_traj_u.flags}")

# real_traj_x = np.asfortranarray(real_traj_x).astype(np.float64)
# real_traj_u = np.asfortranarray(real_traj_u).astype(np.float64)

# num_loop = real_traj_x.shape[0]

# print("=============Start Tracking=============")
# for i_loop in range(num_loop):
#     # print("here2")
#     # plt.figure()

#     # print("Iteration {}".format(i_loop))
#     # print("Initial conditions {}".format(real_traj[i_loop]))

#     # # Solve MLD
#     # t1 = time.time()
#     # # sol = MLD_solver.solve_full_problem(x0_MLD)
#     # sol = MPC_solver.solve_update(i_loop, real_traj[i_loop], real_traj_u[i_loop], False)
#     # if sol['success'] == 0:
#     #     while not sol['success']:
#     #         sol = MPC_solver.solve_update(i_loop, real_traj[i_loop], real_traj_u[i_loop], True)
#     #         print("success: ", sol['success'])
#     #         time.sleep(3)
#     # tc = time.time() - t1
#     # print(colored("Speed " + str(1/(tc)) + " Hz", 'green'))
#     # time_consumed.append(tc)

#     # sol_x = np.zeros([N+1, nx])
#     # for ii in range(N+1):
#     #     for ix in range(nx):
#     #         sol_x[ii, ix] = sol['x_' + str(ii) + '_' + str(ix)]

#     # sol_u = np.zeros([N, nu])
#     # for ii in range(N):
#     #     for iu in range(nu):
#     #         sol_u[ii, iu] = sol['u_' + str(ii) + '_' + str(iu)]   

#     print("here2")
#     plt.figure()

#     print("Iteration {}".format(i_loop))
#     print("Initial conditions {}".format(real_traj_x[i_loop]))

#     # Solve MPC
#     t1 = time.time()
#     # sol = MLD_solver.solve_full_problem(x0_MLD)
#     sol = MPC_solver.solve_update(i_loop, np.asfortranarray(real_traj_x[:i_loop+1]), np.asfortranarray(real_traj_u[:i_loop+1]), False)
#     if sol['success'] == 0:
#         # sol_x = np.zeros([len(real_traj_x), nx])
#         sol_x = real_traj_x
#         # sol_u = np.zeros([len(real_traj_u), nu])
#         sol_u = real_traj_u
#         # while not sol['success']:
#             # sol = MPC_solver.solve_update(i_loop, real_traj_x[i_loop], real_traj_u[i_loop], True)
#             # print("success: ", sol['success'])
#             # time.sleep(3)
#     else:
#         tc = time.time() - t1
#         print(colored("Speed " + str(1/(tc)) + " Hz", 'green'))
#         time_consumed.append(tc)

#         sol_x = np.zeros([N+1, nx])
#         for ii in range(N+1):
#             for ix in range(nx):
#                 sol_x[ii, ix] = sol['x_' + str(ii) + '_' + str(ix)]

#         sol_u = np.zeros([N, nu])
#         for ii in range(N):
#             for iu in range(nu):
#                 sol_u[ii, iu] = sol['u_' + str(ii) + '_' + str(iu)]

#     # sol_u = np.array([sol['control_x'], sol['control_y']])

#     # plot results
#     plot_result(sol_x, sol_u, 'black', '--')
#     plt.grid()
#     plt.savefig(f'figures/MLD_{i_loop}', dpi=400, bbox_inches="tight")

#     # Propagate dynamics
#     # curr_pos = x0_MLD[0:2]; curr_vel = x0_MLD[2:4]
#     # next_pos = curr_pos + curr_vel*dT_dyn
#     # next_vel = curr_vel + sol_u*dT_dyn

#     # x0_MLD = np.array([next_pos[0], next_pos[1], next_vel[0], next_vel[1]])

# print("Solving times are {}".format(time_consumed))
# print("Averaged MLD solving time is {} ms, or {} Hz".format(1000*np.average(np.array(time_consumed)), 1/np.average(np.array(time_consumed))))