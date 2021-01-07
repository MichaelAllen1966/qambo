import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


results_files = {
    'random': 'results_random_action.csv',
    'ddqn': 'results_ddqn.csv',
    '3dqn': 'results_d3qn.csv',
    'noisy 3dqn': 'results_noisy_d3qn.csv',
    'pr 3dqn': 'results_pr_d3qn.csv',
    'pr noisy 3dqn': 'results_pr_noisy_d3qn.csv',
    'bagging ddqn': 'results_bagging_ddqn.csv',
    'bagging 3dqn': 'results_bagging_3dqn.csv',
    'bagging noisy 3dqn': 'results_bagging_noisy_d3qn.csv',
    'bagging pr noisy 3dqn': 'results_bagging_pr_noisy_d3qn.csv'
    }


# Call to response

index = 0

fig = plt.figure(figsize=(14,7))


ax1 = fig.add_subplot(131)
path = './1_incident_points_3_ambo/output/'
x = []
labels = []
for key, value in results_files.items():
    index += 1
    labels.append(str(index) + ': ' + key)
    filename = path + value
    df = pd.read_csv(filename)
    x.append(df['call_to_arrival'].values) 
    
ax1.boxplot(x, whis=1000, widths=0.8)
ax1.set_yticks(np.arange(10,32,2))
ax1.set_ylabel('Call to arrival (minutes)')
ax1.set_title('1 incident area, 3 ambulances')


ax2 = fig.add_subplot(132)
path = './2_incident_points_6_ambo/output/'
x = []
for key, value in results_files.items():
    filename = path + value
    df = pd.read_csv(filename)
    x.append(df['call_to_arrival'].values) 
    
ax2.boxplot(x, whis=1000, widths=0.8)
ax2.set_yticks(np.arange(10,32,2))
ax2.set_ylabel('Call to arrival (minutes)')
ax2.set_title('2 incident areas, 6 ambulances')


ax3 = fig.add_subplot(133)
path = './2_incident_points_6_ambo/output/'
x = []
for key, value in results_files.items():
    filename = path + value
    df = pd.read_csv(filename)
    x.append(df['call_to_arrival'].values) 
    
ax3.boxplot(x, whis=1000, widths=0.8)
ax3.set_yticks(np.arange(10,32,2))
ax3.set_ylabel('Call to arrival (minutes)')
ax3.set_title('3 incident areas, 9 ambulances')


leg = fig.legend(labels, handlelength=0, handletextpad=0, fancybox=True,
                ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.03),
                markerscale=0.0)

for item in leg.legendHandles:
    item.set_visible(False)
    
plt.tight_layout(pad=2)
plt.savefig('call_to_arrival.jpg', dpi=300, bbox_inches='tight')
plt.show()

############################################################################### Assignment to response

index = 0

fig = plt.figure(figsize=(14,7))


ax1 = fig.add_subplot(131)
path = './1_incident_points_3_ambo/output/'
x = []
labels = []
for key, value in results_files.items():
    index += 1
    labels.append(str(index) + ': ' + key)
    filename = path + value
    df = pd.read_csv(filename)
    x.append(df['assign_to_arrival'].values) 
    
ax1.boxplot(x, whis=1000, widths=0.8)
ax1.set_ylim(8,26)
ax1.set_ylabel('Call to response (minutes)')
ax1.set_title('1 incident area, 3 ambulances')


ax2 = fig.add_subplot(132)
path = './2_incident_points_6_ambo/output/'
x = []
for key, value in results_files.items():
    filename = path + value
    df = pd.read_csv(filename)
    x.append(df['assign_to_arrival'].values) 
    
ax2.boxplot(x, whis=1000, widths=0.8)
ax2.set_ylim(8,26)
ax2.set_ylabel('Call to response (minutes)')
ax2.set_title('2 incident areas, 6 ambulances')


ax3 = fig.add_subplot(133)
path = './2_incident_points_6_ambo/output/'
x = []
for key, value in results_files.items():
    filename = path + value
    df = pd.read_csv(filename)
    x.append(df['assign_to_arrival'].values) 
    
ax3.boxplot(x, whis=1000, widths=0.8)
ax3.set_ylim(8,26)
ax3.set_ylabel('Call to response (minutes)')
ax3.set_title('3 incident areas, 9 ambulances')


leg = fig.legend(labels, handlelength=0, handletextpad=0, fancybox=True,
                ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.03),
                markerscale=0.0)

for item in leg.legendHandles:
    item.set_visible(False)
    
plt.tight_layout(pad=2)
plt.savefig('assign_to_arrival.jpg', dpi=300, bbox_inches='tight')
plt.show()