import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pdb

def window_smooth(signal_in):
    signal = signal_in

    window_size = 3
    th = int((window_size-1)/2)

    for ii in range(len(signal)):
        if ii >= th and ii <= len(signal)-th-1:
            signal[ii] = np.average(signal[(ii-th):(ii+th)])

    return signal

BD_ws_10 = sio.loadmat('t_spend_Benders_100sec_N10_cpp.mat')
Gurobi_10 = sio.loadmat('t_spend_Gurobi_100sec_N10.mat')
BD_ws_15 = sio.loadmat('t_spend_Benders_100sec_N15_cpp.mat') 
Gurobi_15 = sio.loadmat('t_spend_Gurobi_100sec_N15.mat')
# BD_ws_20 = sio.loadmat('t_spend_Benders_100sec_N20')

delta_T_dyn = 0.005

time_BD_10 = BD_ws_10['time_Benders'][0]; time_Gurobi_10 = Gurobi_10['time_Gurobi'][0]
opt_cuts_BD_10 = np.concatenate((np.array([0]), BD_ws_10['opt_cuts_traj'][0])); feas_cuts_BD_10 = np.concatenate((np.array([0]), BD_ws_10['feas_cuts_traj'][0]))
cost_GBD_10 = BD_ws_10['cost_Benders'][0]; cost_Gurobi_10 = Gurobi_10['cost_Gurobi'][0]
iter_traj_GBD_10 = BD_ws_10['num_iter_traj'][0]
time_x_Benders_10 = delta_T_dyn*np.linspace(0, len(BD_ws_10['time_traj'][0])-1, len(BD_ws_10['time_traj'][0]))
time_x_Gurobi_10 = delta_T_dyn*np.linspace(0, len(Gurobi_10['time_traj'][0])-1, len(Gurobi_10['time_traj'][0]))
Hz_BD_10 = 1/time_BD_10; Hz_Gurobi_10 = 1/time_Gurobi_10
NN_10 = BD_ws_10['N'].squeeze().item()

time_BD_15 = BD_ws_15['time_Benders'][0]; time_Gurobi_15 = Gurobi_15['time_Gurobi'][0]
opt_cuts_BD_15 = np.concatenate((np.array([0]), BD_ws_15['opt_cuts_traj'][0])); feas_cuts_BD_15 = np.concatenate((np.array([0]), BD_ws_15['feas_cuts_traj'][0]))
cost_GBD_15 = BD_ws_15['cost_Benders'][0]; cost_Gurobi_15 = Gurobi_15['cost_Gurobi'][0]
iter_traj_GBD_15 = BD_ws_15['num_iter_traj'][0]
time_x_Benders_15 = delta_T_dyn*np.linspace(0, len(BD_ws_15['time_traj'][0])-1, len(BD_ws_15['time_traj'][0]))
time_x_Gurobi_15 = delta_T_dyn*np.linspace(0, len(Gurobi_15['time_traj'][0])-1, len(Gurobi_15['time_traj'][0]))
Hz_BD_15 = 1/time_BD_15; Hz_Gurobi_15 = 1/time_Gurobi_15
NN_15 = BD_ws_15['N'].squeeze().item()
NN_15 = 15

# time_avg_10 = np.average(time_BD_10)
# time_BD_sub_avg_10 = np.average(BD_ws_10['time_Benders_real'][0] - BD_ws_10['time_Benders_master'][0])
# time_BD_master_avg_10 = np.average(BD_ws_10['time_Benders_master'][0])
# print("For N={}, Subproblem percentage {} %".format(NN_10, 100*time_BD_sub_avg_10/time_avg_10))
# print("For N={}, Master problem percentage {} %".format(NN_10, 100*time_BD_master_avg_10/time_avg_10))

# time_avg_15 = np.average(time_BD_15)
# time_BD_sub_avg_15 = np.average(BD_ws_15['time_Benders_real'][0] - BD_ws_15['time_Benders_master'][0])
# time_BD_master_avg_15 = np.average(BD_ws_15['time_Benders_master'][0])
# print("For N={}, Subproblem percentage {} %".format(NN_15, 100*time_BD_sub_avg_15/time_avg_15))
# print("For N={}, Master problem percentage {} %".format(NN_15, 100*time_BD_master_avg_15/time_avg_15))

# time_avg_20 = np.average(BD_ws_20['time_Benders'][0])
# time_BD_sub_avg_20 = np.average(BD_ws_20['time_Benders_real'][0] - BD_ws_20['time_Benders_master'][0])
# time_BD_master_avg_20 = np.average(BD_ws_20['time_Benders_master'][0])
# print("For N={}, Subproblem percentage {} %".format(20, 100*time_BD_sub_avg_20/time_avg_20))
# print("For N={}, Master problem percentage {} %".format(20, 100*time_BD_master_avg_20/time_avg_20))

# For N=10, Subproblem percentage 39.89256947253139 %
# For N=10, Master problem percentage 13.905109007550633 %
# For N=15, Subproblem percentage 44.28026329483789 %
# For N=15, Master problem percentage 16.82691410008278 %

# max_id = 1400
max_id = 600

plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(time_x_Benders_10[:max_id], Hz_BD_10[:max_id], label='GBD', color='blue', linewidth=2.0)
plt.plot(time_x_Gurobi_10[:max_id], Hz_Gurobi_10[:max_id], label='Gurobi', color='green', linewidth=2.0)
plt.grid()
plt.legend(loc='upper right')
plt.ylabel('Solving speed/Hz', fontsize=20)
plt.title('N={}'.format(NN_10), fontsize=20, fontweight="bold")
ax1.xaxis.label.set_fontweight('bold')
ax1.yaxis.label.set_fontweight('bold')
ax1.xaxis.set_tick_params(width=5)
ax1.yaxis.set_tick_params(width=5)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)

ax2 = plt.subplot(3, 1, 2)
plt.plot(time_x_Benders_10[:max_id], opt_cuts_BD_10[:max_id], label='Opt. cuts', color='red', linewidth=2.5)
plt.plot(time_x_Benders_10[:max_id], feas_cuts_BD_10[:max_id], label='Feas. cuts', color='black', linewidth=2.5)
plt.grid()
plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
plt.legend(loc='upper right')
plt.ylabel('# of stored cuts', fontsize=20)
ax2.xaxis.label.set_fontweight('bold')
ax2.yaxis.label.set_fontweight('bold')
ax2.xaxis.set_tick_params(width=5)
ax2.yaxis.set_tick_params(width=5)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax3 = plt.subplot(3, 1, 3)
plt.plot(time_x_Benders_10[:max_id], iter_traj_GBD_10[:max_id], label='# of Benders iteration', color='black', linewidth=2.5)
plt.grid()
plt.yticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
# plt.legend(loc='upper right')
plt.ylabel('# of GBD iterations', fontsize=20)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
ax3.xaxis.label.set_fontweight('bold')
ax3.yaxis.label.set_fontweight('bold')
ax3.xaxis.set_tick_params(width=5)
ax3.yaxis.set_tick_params(width=5)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)

plt.figure(2)
ax4 = plt.subplot(3, 1, 1)
plt.plot(time_x_Benders_15[:max_id], Hz_BD_15[:max_id], label='GBD', color='blue', linewidth=2.0)
plt.plot(time_x_Gurobi_15[:max_id], Hz_Gurobi_15[:max_id], label='Gurobi', color='green', linewidth=2.0)
plt.grid()
plt.legend(loc='upper right')
plt.ylabel('Solving speed/Hz', fontsize=20)
plt.title('N={}'.format(NN_15), fontsize=20, fontweight="bold")
ax4.xaxis.label.set_fontweight('bold')
ax4.yaxis.label.set_fontweight('bold')
ax4.xaxis.set_tick_params(width=5)
ax4.yaxis.set_tick_params(width=5)
ax4.tick_params(axis='x', labelsize=15)
ax4.tick_params(axis='y', labelsize=15)

ax5 = plt.subplot(3, 1, 2)
plt.plot(time_x_Benders_15[:max_id], opt_cuts_BD_15[:max_id], label='Opt. cuts', color='red', linewidth=2.5)
plt.plot(time_x_Benders_15[:max_id], feas_cuts_BD_15[:max_id], label='Feas. cuts', color='black', linewidth=2.5)
plt.grid()
plt.yticks([0, 100, 200, 300, 400, 500, 600])
plt.legend(loc='upper right')
plt.ylabel('# of stored cuts', fontsize=20)
ax5.xaxis.label.set_fontweight('bold')
ax5.yaxis.label.set_fontweight('bold')
ax5.xaxis.set_tick_params(width=5)
ax5.yaxis.set_tick_params(width=5)
ax5.tick_params(axis='x', labelsize=15)
ax5.tick_params(axis='y', labelsize=15)

ax6 = plt.subplot(3, 1, 3)
plt.plot(time_x_Benders_15[:max_id], iter_traj_GBD_15[:max_id], label='# of Benders iteration', color='black', linewidth=2.5)
plt.grid()
plt.yticks([1, 10, 20, 30, 40, 50])
# plt.legend(loc='upper right')
plt.ylabel('# of GBD iterations', fontsize=20)
plt.xlabel('time/sec', fontsize=20)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
ax6.xaxis.label.set_fontweight('bold')
ax6.yaxis.label.set_fontweight('bold')
ax6.xaxis.set_tick_params(width=5)
ax6.yaxis.set_tick_params(width=5)
ax6.tick_params(axis='x', labelsize=15)
ax6.tick_params(axis='y', labelsize=15)

plt.show()

