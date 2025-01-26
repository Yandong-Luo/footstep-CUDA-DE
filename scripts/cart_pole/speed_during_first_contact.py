import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pdb

BD_ws = sio.loadmat('t_spend_Benders_1sec_cpp.mat')
Gurobi = sio.loadmat('t_spend_Gurobi_1sec.mat')
BD_no_ws = sio.loadmat('t_spend_Benders_1sec_cpp_no_ws.mat')

delta_T_dyn = 0.005

time_BD = BD_ws['time_Benders'][0]; time_Gurobi = Gurobi['time_Gurobi'][0]; time_BD_no_ws = BD_no_ws['time_Benders'][0]
opt_cuts_BD = np.concatenate((np.array([0]), BD_ws['opt_cuts_traj'][0])); feas_cuts_BD = np.concatenate((np.array([0]), BD_ws['feas_cuts_traj'][0]))
cost_GBD = BD_ws['cost_Benders'][0]; cost_Gurobi = Gurobi['cost_Gurobi'][0]
time_x_Benders = delta_T_dyn*np.linspace(0, len(BD_ws['time_traj'][0])-1, len(BD_ws['time_traj'][0]))
time_x_Gurobi = delta_T_dyn*np.linspace(0, len(Gurobi['time_traj'][0])-1, len(Gurobi['time_traj'][0]))
time_x_Benders_no_ws = delta_T_dyn*np.linspace(0, len(BD_no_ws['time_traj'][0])-1, len(BD_no_ws['time_traj'][0]))

Hz_BD = 1/time_BD
Hz_Gurobi = 1/time_Gurobi
Hz_BD_no_ws = 1/time_BD_no_ws

max_id = 40
ax1 = plt.subplot(2, 2, 1)
plt.plot(1000*time_x_Benders[:max_id], Hz_BD[:max_id], label='GBD with warm-start', color='blue', linewidth=2.5)
plt.plot(1000*time_x_Gurobi[:max_id], Hz_Gurobi[:max_id], label='Gurobi', color='green', linewidth=2.5)
plt.plot(1000*time_x_Benders_no_ws[:max_id], Hz_BD_no_ws[:max_id], label='GBD w/o warm-start', color='orange', linewidth=2.5)
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('time/ms', fontsize=20)
plt.ylabel('Solving speed/Hz', fontsize=20)
# plt.title('Solving speed during first contact')
ax1.xaxis.label.set_fontweight('bold')
ax1.yaxis.label.set_fontweight('bold')
ax1.xaxis.set_tick_params(width=5)
ax1.yaxis.set_tick_params(width=5)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)

ax2 = plt.subplot(2, 2, 2)
plt.plot(1000*time_x_Benders[:max_id], cost_GBD[:max_id], label='GBD with warm-start', color='blue', linewidth=2.5)
plt.plot(1000*time_x_Gurobi[:max_id], cost_Gurobi[:max_id], label='Gurobi', color='green', linewidth=2.5)
plt.grid()
plt.legend()
plt.ylim([0, 10000])
plt.xlabel('time/ms', fontsize=20)
plt.ylabel('Cost', fontsize=20)
# plt.title('Objective function value')
ax2.xaxis.label.set_fontweight('bold')
ax2.yaxis.label.set_fontweight('bold')
ax2.xaxis.set_tick_params(width=5)
ax2.yaxis.set_tick_params(width=5)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax3 = plt.subplot(2, 2, 3)
plt.plot(1000*time_x_Benders[:max_id], BD_ws['num_iter_traj'][0][:max_id], label='Number of iterations for GBD with warm-start', color='black', linewidth=2.5)
plt.grid()
plt.legend()
plt.xlabel('time/ms', fontsize=20)
plt.ylabel('# of GBD iterations', fontsize=20)
# plt.title('# of iterations')
ax3.xaxis.label.set_fontweight('bold')
ax3.yaxis.label.set_fontweight('bold')
ax3.xaxis.set_tick_params(width=5)
ax3.yaxis.set_tick_params(width=5)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)

ax4 = plt.subplot(2, 2, 4)
plt.plot(1000*time_x_Benders[:max_id], BD_ws['feas_cuts_traj'][0][:max_id], label='GBD Feasibility cuts', color='black', linewidth=2.5)
plt.plot(1000*time_x_Benders[:max_id], BD_ws['opt_cuts_traj'][0][:max_id], label='GBD Optimality cuts', color='red', linewidth=2.5)
plt.grid()
plt.legend()
plt.xlabel('time/ms', fontsize=20)
plt.ylabel('# of stored cuts', fontsize=20)
# plt.title('# of stored cuts during contact')
ax4.xaxis.label.set_fontweight('bold')
ax4.yaxis.label.set_fontweight('bold')
ax4.xaxis.set_tick_params(width=5)
ax4.yaxis.set_tick_params(width=5)
ax4.tick_params(axis='x', labelsize=15)
ax4.tick_params(axis='y', labelsize=15)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.2, hspace=0.2)

plt.show()
