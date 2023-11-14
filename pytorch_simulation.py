import torch
import weight
import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)
image = cv2.imread('107.bmp',0)
image = np.array(image, dtype=int) - 128 - 24
conv_weight = np.array(weight.weights)
input = image[0:256:4,0:256:4][np.newaxis,:]
cc = torch.nn.ZeroPad2d((0,3,0,3))

tensor_input = cc(torch.tensor(input, dtype=float))


tensor_weight = torch.tensor(conv_weight, dtype=float)
result = torch.nn.functional.conv2d(tensor_input, tensor_weight, stride = 2)

result = result / 16
maxpool = torch.nn.MaxPool2d(4)
result = maxpool(result)


result = torch.tensor(torch.flatten(result), dtype=torch.float32)
fc1 = torch.tensor(np.array(weight.fc_weights_1), dtype=torch.float32)
m1 = torch.nn.Linear(1024,16)
m1.weight = torch.nn.Parameter(fc1)
result = m1(result)
print(result)
'''
fc1 = torch.tensor(np.array(weight.fc_weights_2), dtype=torch.float32)
m2 = torch.nn.Linear(16,3)
m2.weight = torch.nn.Parameter(fc1)
print(m2(result) / 1024)
'''