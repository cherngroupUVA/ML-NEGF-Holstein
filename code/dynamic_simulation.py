import torch
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import os

np.random.seed(4785447)
size_l = 40
size_w = 30
window_boundary = 6 
cutoff = 6 #cutoff radius
ramp = 18 
input_num_feature = 113 # number of input features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_input = "./data/"
dir_out = './simulation/'

diff_Q = np.zeros((450, size_w, (size_l - 2 * window_boundary)))





class holstein_dynamics:

    def __init__(self, model, neighbor_2d, Q, neighbor_list, ooc, P_dis, parameters):
        
       
        
        self.kT = 0.1 
        self.dt = 0.1
        
        self.g = 1.5
        self.kkon = 1.0
        self.kapnn = 0.18
        self.damping = 0.2
        
        self.Ns = size_l * size_w
        
        self.rand_var = np.sqrt(2.0 * self.kT * self.dt / self.damping)
        self.net = model
        self.Q = Q
        self.neighbor_2d = neighbor_2d
        self.neighbor_list = neighbor_list
        self.parameters_Q = parameters[0]
        self.parameters_n = parameters[1]
        
        if ooc is not None:
            self.ooc = ooc
        else:
            self.ooc = torch.zeros(((size_l - 2 *window_boundary) * size_w)).float().to(device)
            
        if P_dis is not None:
            self.P_dis = P_dis
        else:
            self.P_dis = torch.zeros(self.Ns).float().to(device)
            
        
        
            
    def Find_Features(self,neighbor_spin):
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
        

    def create_bond_matrix(self, spin_tensor, neighbor_list):
        bond_feature_for_all = []
        for i in range(size_l * size_w):                     
            x = i % size_l
            if x < window_boundary or x > size_l - window_boundary - 1:
                continue
            else:
                index = (ramp+1) * i
                #print("This is i: ", i)
                #print("This is index: ", index)
                neighbor_spin = []
                for p in range(index, index + ramp + 1):
                    for q in range(len(neighbor_list[p])):
                        position = int(neighbor_list[p][q])
                        neighbor_spin.append([position, spin_tensor[position]])
                bond_feature_for_all.append(self.Find_Features(neighbor_spin))
        return  torch.FloatTensor(bond_feature_for_all).float().to(device)
    
    
    def screenANDscale_tensor2(self, IR_tensor, tensor_max):
        tensor_min = -1 * tensor_max
        return 2 * (IR_tensor - tensor_min) / (tensor_max - tensor_min) - 1

        
    def Anti_screen2(self, IR_tensor, tensor_max):
        tensor_min = -1 * tensor_max
        return 1/2 * (IR_tensor + 1) * (tensor_max - tensor_min) + tensor_min
    
    
    
    def step(self, file_num):
        

        if (file_num <= 547):  
            original_Q = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(file_num)+'.dat',usecols=(5))) # Q value at current step
            original_n = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(file_num)+'.dat',usecols=(2))) # n value at current step
        
        else:
            original_Q = torch.tensor(np.loadtxt(data_input+'t547_1094/cq'+str(file_num - 547)+'.dat',usecols=(5)))
            original_n = torch.tensor(np.loadtxt(data_input+'t547_1094/cq'+str(file_num - 547)+'.dat',usecols=(2)))
            
        if (file_num <= 546):
            original_Q_next = torch.tensor(np.loadtxt(data_input+'t0_547/cq'+str(file_num+1)+'.dat',usecols=(5))) # Q value at next step
        else:
            original_Q_next = torch.tensor(np.loadtxt(data_input+'t547_1094/cq'+str(file_num - 547 + 1)+'.dat',usecols=(5)))
           
        self.Q = self.screenANDscale_tensor2(self.Q,self.parameters_Q).float().to(device)
        
        
        self.P_dis = torch.zeros(self.Ns).float().to(device) 
        
        input_features = self.create_bond_matrix(self.Q.tolist(), self.neighbor_list)
        self.ooc = self.net(input_features).float().to(device)  
        
        self.ooc = self.Anti_screen2(self.ooc, self.parameters_n)  
        den_local = self.ooc.reshape(size_w,size_l - 2 * window_boundary)
        self.Q = self.Anti_screen2(self.Q,self.parameters_Q)
        
    
        for i in range(0,self.Ns):
            x = i % size_l
            if (0 <= x < window_boundary or x > size_l - window_boundary - 1):
                continue

            y = int (i / size_l)
            x2 = x - window_boundary 
           
            self.P_dis[i] = -1 * self.g * (den_local[y,x2] - 0.5) - self.kkon * self.Q[i]
            
            # now is the second part of the elastic force; like the 2nd item in eq 5:
                
            for k in range(0,4): 
                if self.neighbor_2d[i][k][1] == -1:
                    continue
                else:
                    index = self.neighbor_2d[i][k][0]
                    if self.neighbor_2d[i][k][1] == 0:
                        neighbor_Q = original_Q[index]
                    else:
                        neighbor_Q = self.Q[index]
                self.P_dis[i] = self.P_dis[i] - self.kapnn * neighbor_Q
         
        
        for i in range(0, self.Ns):
            x = i % size_l
            if (0 <= x < window_boundary or x > size_l - window_boundary - 1):
                self.Q[i] = original_Q_next[i]
                
            else:
                rd = np.random.normal(0,1)
                self.Q[i] = self.Q[i] + self.P_dis[i] * self.dt / self.damping + rd[i] * self.rand_var
            
    
        self.ooc = self.ooc.detach().numpy()
        
       
        
                      
               
        
            
                
                
                
                
                
                
                
                
                
                
                
    
            
        
        
        
        
        
        
        
        # for i in range(self.ts):
        #     self.Q[i] += self.velocity[i] * self.dt + 0.5 * (self.force[i] / self.mass) * self.dt * self.dt

        # force_prev = self.force
        # self.force = self.calc_force(self.Q)
        # self.occ = self.calc_occ(self.Q)

        # for i in range(self.ts):
        #     self.velocity[i] += 0.5 * self.dt * (self.force[i] + force_prev[i]) / self.mass
        # for i in range(self.ts):
        #     self.velocity[i] = self.a_x * self.velocity[i] + self.b_x * math.sqrt(self.kT) * np.random.randn()
        
        
        
        # for p in range(0, size_w):
        #     for q in range(0, size_l - 2 * window_boundary):
        #         diff_Q[file_num - 70, p, q] = original_Q_next[p * size_l + q + window_boundary] - self.Q[p * size_l + q + window_boundary]
        
        
        #print("This is at: ",file_num, "  with mean: ", np.mean(np.array(diff_Q.reshape(-1))))
                
            
        # for i in range(0, self.Ns):
        #     rd = np.random.normal(0.0, 1.0)  # how to control it????  
        #     self.Q[i] = original_Q_next[i] +  rd * self.rand_var 
            
        #rd = np.random.normal(0.0, 1.0)    
        #self.Q = original_Q_next #+ rd * self.rand_var









