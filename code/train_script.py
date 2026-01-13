# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:43:39 2024

@author: 14026
"""

import torch
import torch.nn.functional as f
import torch.utils.data as data
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
import tarfile
import torch.optim.lr_scheduler as lr_scheduler  # should have a larger patience!!!!!!
import time

import matplotlib.pyplot as plt
import os
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import pylab
from pylab import *
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

############################Parameters################################

size_l = 40
size_w = 30
window_boundary = 6 
cutoff = 6 #cutoff radius
from_bench_mark =False
epoch_start = 0
duration = 1000
ramp = 18 
input_num_feature = 113 # number of input features
neighbor_input = "./neighbor/"
data_input = "./data/"

from_bench_mark = False






#######################Train Begins####################################################  

             
                 
class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(input_size, 4096)
        self.hidden_1 = torch.nn.Linear(4096, 2048)
        self.hidden_2 = torch.nn.Linear(2048, 1024)
        self.hidden_3 = torch.nn.Linear(1024, 512)
        self.hidden_4 = torch.nn.Linear(512, 256)
        self.hidden_5 = torch.nn.Linear(256, 128)
        self.hidden_6 = torch.nn.Linear(128, 64)
        self.hidden_7 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.input(x))
        x = torch.nn.functional.relu(self.hidden_1(x))
        x = torch.nn.functional.relu(self.hidden_2(x))
        x = torch.nn.functional.relu(self.hidden_3(x))
        x = torch.nn.functional.relu(self.hidden_4(x))
        x = torch.nn.functional.relu(self.hidden_5(x))
        x = torch.nn.functional.relu(self.hidden_6(x))
        x = torch.nn.functional.relu(self.hidden_7(x))
        x = self.output(x)
        return x



def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)

def create_bond_matrix(spin_tensor, neighbor_list):
    bond_feature_for_all = []
    for i in range(size_l * size_w):                         
        x = i % size_l
        if x < window_boundary or x > size_l - window_boundary - 1:
            continue
            
        else:
            index = (ramp+1) * i
            
            neighbor_spin = []
            for p in range(index, index + ramp + 1): 
                for q in range(len(neighbor_list[p])): 
                    position = int(neighbor_list[p][q])
                    neighbor_spin.append([position, spin_tensor[position]])
            bond_feature_for_all.append(Find_Features(neighbor_spin))
    return  torch.FloatTensor(bond_feature_for_all).float().to(device)

def Find_Features(neighbor_spin): # This function create the generalized coordinates
    features = []
    ref_upper = 0
    ref_lower = 0
    center = neighbor_spin[0][0]
    center_x = center % size_l
    center_y = (int)((center - center_x) / size_l )
    all_detail = []
    all_detail_level = []
    for i in range(0, len(neighbor_spin)):
        single_detail = []
        item = neighbor_spin[i][1]
        position = neighbor_spin[i][0]
        position_x = position % size_l
        position_y = (int)((position - position_x) / size_l )
        if (np.abs(position_y - center_y) > cutoff):
            if ((position_y - center_y) > cutoff) : 
                position_y = position_y - size_w
            elif (center_y - position_y > cutoff):
                position_y = position_y + size_w
        single_detail.append(position_x)
        single_detail.append(position_y - center_y)
        single_detail.append(item)
        
        if (position_y == center_y):
            all_detail_level.append(single_detail)
        else:
            all_detail.append(single_detail)
       
        if (position_y < center_y):
            ref_lower = ref_lower + item
        elif (position_y > center_y):
            ref_upper = ref_upper + item
        else:
            ref_lower = ref_lower + item / 2
            ref_upper = ref_upper + item / 2
        
    ref_v = (ref_upper - ref_lower) / 2
    if (ref_v !=0):
        ref_v = ref_v / np.abs(ref_v)
    all_detail.sort(key = lambda x: x[0])
    count = 0
    for j in range(len(all_detail_level)):
        features.append(all_detail_level[j][2])
    while (count < len(all_detail) - 1):
        if (all_detail[count][1] == -1 * all_detail[count + 1][1]):
            ir_iv = (all_detail[count][2] + all_detail[count + 1][2]) / 2
            ir_v = (all_detail[count][2] - all_detail[count + 1][2]) / 2
            features.append(ir_iv)
            features.append(ir_v * ref_v)
            count = count + 2
        else:
            print("something not right: ", all_detail[count][1], "  ", all_detail[count + 1][1])
            break

    return features



def create_force_matrix(force_tensor):
    bond_feature_for_all = []
    for i in range(size_l * size_w):
        x = i % size_l
        if x < window_boundary or x > size_l - window_boundary - 1:
            continue
    
        bond_feature_for_all.append(force_tensor[i].item())
    return torch.tensor(bond_feature_for_all).float().to(device)

def screenANDscale_tensor2(IR_tensor, tensor_max):
    tensor_min = -1 * tensor_max
    return 2 * (IR_tensor - tensor_min) / (tensor_max - tensor_min) - 1
    

##################################LOAD NEIGHBOR INFORMATION#################################################


with open(neighbor_input + "40_30_ramp_18.csv", 'r') as file:
    reader = csv.reader(file)
    neighbor_list = list(reader)
    

start = 70

config = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(start)+'.dat',usecols=(2,5))).reshape(1,-1,2)

config_test = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(72)+'.dat',usecols=(2,5))).reshape(1,-1,2)


for i in range(start + 1, 160): 
    if (i+1) % 3 != 0:
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
    else:
        config_1_test = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config_test = torch.cat((config_test, config_1_test),dim=0)
        
    
    
for i in range(350, 521): 

    if (i+1) % 3 != 0:
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
    else:
        config_1_test = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config_test = torch.cat((config_test, config_1_test),dim=0)
    
for i in range(160, 351): #in order to emphasize those snapshots in the middle of evolution 
    if (i+1) % 3 != 0:
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
        config_1 = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config = torch.cat((config, config_1),dim=0)
        
        
    else:
        config_1_test = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(i)+'.dat',usecols=(2,5))).reshape(1,-1,2)
        config_test = torch.cat((config_test, config_1_test),dim=0)
    
    
    
    
Q = config[::,::,1].float().to(device)
n = config[::,::,0].float().to(device)

#print("This is Q's shape: ",  Q.shape)

parameters=[]
parameters.append(max(np.abs(torch.max(Q).item()), np.abs(torch.min(Q).item())))
parameters.append(max(np.abs(torch.max(n).item()), np.abs(torch.min(n).item())))


with open('T_repeated_middle_para.csv','a',newline='') as file:
    writer = csv.writer(file)
    writer.writerows([parameters]) 



indices = []

for i in range(0, Q.shape[0]):
    if (i> -1):
        indices.append(i)
    
indices = torch.tensor(indices).to(device)

dim = 0



Q_train = screenANDscale_tensor2(torch.index_select(Q,dim,indices),parameters[0]).float().to(device)

n_train = screenANDscale_tensor2(torch.index_select(n,dim,indices),parameters[1]).float().to(device)


torch_data_set_train = data.TensorDataset(Q_train,n_train)
loader_train = data.DataLoader(dataset=torch_data_set_train, batch_size=1, shuffle=True)

net = Net(input_num_feature).float()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=4, threshold=1e-7, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-8)
loss_func = torch.nn.MSELoss()



if from_bench_mark:
    print("Now loading the previous model")
    prev_model_data = torch.load("./model/r8_14.pt")               
    net.load_state_dict(prev_model_data['model'])
    optimizer.load_state_dict(prev_model_data['optimizer'])
else:
    print("xavier normal init")
    net.apply(init_normal)

loss_func = torch.nn.MSELoss()


saved_loss = 100000
time_start = time.time()
for epoch in range(epoch_start,epoch_start+duration): #Train begins
    fold = np.random.randint(0,10)            
    loss_perstep = []
    
    for step, (Q,n) in enumerate(loader_train):
        lattice_Qs = Q[0]
        lattice_ns = n[0]
        input_features = create_bond_matrix(lattice_Qs.tolist(), neighbor_list)
        out_features = create_force_matrix(lattice_ns)
        optimizer.zero_grad()
        predicted_ns = net(input_features)
        predicted_ns = predicted_ns.reshape(-1)
        loss = loss_func(out_features,predicted_ns)
        loss_perstep.append(loss.item())
        loss.backward()
        optimizer.step()
        
        
    average_loss = sum(loss_perstep) / len(loss_perstep)
    #print("This is average loss: ", average_loss)
    outfile = open("./model.txt", "a")
    outfile.write("epoch: " + str(epoch) + "| time = " + str(time.time() - time_start) ) 
    outfile.write("\n")
    outfile.write("***************\n")
    outfile.close()
    
    
    
    if (average_loss < saved_loss ):                                   #((epoch+1)%100)==0: or (epoch+1) % 1000 == 0
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/model"+".pt")
        saved_loss = average_loss
        best_epoch = []
        best_epoch.append(epoch)
        with open('model.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([best_epoch])    
    
   
    scheduler.step(average_loss)
    info_per_epoch=[]
    info_per_epoch.append(epoch)
    info_per_epoch.append(optimizer.param_groups[-1]['lr'])
    info_per_epoch.append(average_loss)
    with open('model.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([info_per_epoch])    

