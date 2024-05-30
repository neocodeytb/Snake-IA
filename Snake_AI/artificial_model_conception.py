import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

model = nn.Sequential(

    nn.Linear(6,36),
    nn.ReLU(),

    nn.Linear(36,18),
    nn.ReLU(),

    nn.Linear(18,4),

)

ongoing_loss = []
lossfun = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

for epochi in range(1000):
    #Gauche,Droite,Haut,Bas (Dangers), Gauche,Droite,Haut,Bas (Bouffe)


    input = [0,0,0,0,0,0,0,0]
    rdm = random.randint(0,3)
    rdm2 = random.randint(4,7)
    input[rdm] = 1
    input[rdm2] = 1

    if input[4] == 1:
        bouffe = "Gauche"
    if input[5] == 1:
        bouffe = "Droite"
    if input[6] == 1:
        bouffe = "Haut"
    if input[7] == 1:
        bouffe = "Bas"

    if input[0] == 1:
        danger = "Gauche"
    if input[1] == 1:
        danger = "Droite"
    if input[2] == 1:
        danger = "Haut"
    if input[3] == 1:
        danger = "Bas"

    output = model(torch.tensor(input).float())
    action = torch.argmax(output)
    value = torch.max(output)

    if action == 0:
        a = "Gauche"
    elif action == 1:
        a = "Droite"
    elif action == 2:
        a = "Haut"
    elif action == 3:
        a = "Bas"

    if rdm == action:
        reward = -1
    else:
        if rdm2 - 4 == action:
            reward = 1
        else:
            reward = 0

    loss = lossfun(value,torch.tensor(reward).float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ongoing_loss.append(loss.item())

    print(f"Output : {output}, action : {a}, danger : {danger}, bouffe : {bouffe}, reward : {reward} ")

plt.plot(ongoing_loss)
plt.show()