import torch
import torch.nn.functional as F
import math

t1 = torch.Tensor( [1.0/3, 2.0/3] ).float()
t2 = torch.Tensor( [2.0/5 ,3.0/5] ).float()

print(t1 * t2)

print( (t1 *(t1.log()-t2.log())).sum() )
# print(t1 * (t1.log() - t2.log())).sum()

# print(F.kl_div(t2, t1))

'''
a1 = [1.0/3.0, 2.0/3.0]
a2 = [2.0/5.0, 3.0/5.0]

sum = 0.0
for i in range(2):
    print(a1[i]/a2[i], math.log(a1[i]/a2[i]))
    a = a1[i] * math.log(a1[i]/a2[i])
    print(a)
print(sum)
'''

'''
t1 = torch.tensor([t[0][0],t[1][0]]).float()
t2 = torch.tensor([t[0][1], t[1][1]]).float()
print(F.softmax(t1, dim=0))
print(F.softmax(t2, dim=0))

t_sm = F.softmax(t, dim=0)
print(t_sm[:, 0])
print(t_sm[:, 1])

t_sm1 = F.softmax(t, dim=1)
print(t_sm1[0])
print(t_sm1[1])
'''