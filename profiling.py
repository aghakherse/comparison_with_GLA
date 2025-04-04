import time
import torch.nn.functional as F

import math
import torch
from torch import cuda
from torch.autograd import gradcheck
from Linear_Attention import FASTMultiHeadAttention
import fastmax_cuda
import numpy as np
from gla import GatedLinearAttention

LA_ours = FASTMultiHeadAttention() # ours linear attention implementation
LA_torch = FASTMultiHeadAttention(False) # linear attention implemented using pytorch



# look here
b = 4 # batch
h = 16 # head
d = 128 # dimension per head (i.e. embedding_dimension/h)

# n changes from 10^strt to 10^endd. The number of test points are count
rep = 100
count = 10
strt = 3 # log scale
endd = 5 # log scale
# lengths = [1024,4096,8192]


dtype = torch.float32
print("bhd = ",b,",",h,",",d,",")

our_LA_time = np.zeros(count)
gla_time = np.zeros(count)
our_LA_memory = np.zeros(count)
gla_memory = np.zeros(count)
device = torch.device(0)
mask = False


j = -1
print("Our LA Implementation")
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            # print(ii)
            torch.cuda.empty_cache()
            q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            start_time = time.time()
            e = LA_ours(q,k,v,mask)
            # print(e)
            cuda.synchronize()
            end_time = time.time()
            our_LA_time[j] += (end_time - start_time)/rep
        our_LA_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))
        our_LA_time[j] = -1
        our_LA_memory[j] = -1



print("############################################")
print("Gated LA Implementation")
gla = GatedLinearAttention(d = d, h = h, device = torch.device('cuda'))
j = -1
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            torch.cuda.empty_cache()
            x = torch.normal(0,1,[b,int(i),h*d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            start_time = time.time()
            e = gla(x)
            cuda.synchronize()
            end_time = time.time()
            
            gla_time[j] += (end_time - start_time)/rep
        gla_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        gla_time[j] = -1
        gla_memory[j] = -1
        print("OOM for token length of ", int(i))


temp = "["
for i in our_LA_time: temp += str(i) + ", "
temp += "]"
print("Our LA Time = ", temp)
temp = "["
for i in our_LA_memory: temp += str(i) + ", "
temp += "]"
print("Our LA Memory = ", temp)

print()
temp = "["
for i in gla_time: temp += str(i) + ", "
temp += "]"
print("Gated LA Time = ", temp)
temp = "["
for i in gla_memory: temp += str(i) + ", "
temp += "]"
print("Gated LA Memory = ", temp)
