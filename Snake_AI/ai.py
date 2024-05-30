import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def get_stronger_state(snake,board,l,h):
    state = np.empty(0)
    idx = -1
    for ligne in board:
        for element in ligne:
            idx += 1
            if snake.x*h + snake.y == idx:
                state = np.append(state,0)
                state = np.append(state,0)
                state = np.append(state,1)
            else:
                if element == 0:
                    state = np.append(state,0)
                    state = np.append(state, 0)
                    state = np.append(state, 0)
                if element == 1 or element == 3:
                    state = np.append(state,1)
                    state = np.append(state, 0)
                    state = np.append(state, 0)
                if element == 2:
                    state = np.append(state,0)
                    state = np.append(state, 1)
                    state = np.append(state, 0)

    food = np.where(board == 2)
    if food[0].size == 1:

        food_x = food[0][0]
        food_y = food[1][0]

        dx = food_x - snake.x
        dy = food_y - snake.y

        if dx > 0:
            state = np.append(state,1)
            state = np.append(state, 0)
        elif dx < 0:
            state = np.append(state, 0)
            state = np.append(state, 1)
        else:
            state = np.append(state, 0)
            state = np.append(state, 0)

        if dy > 0:
            state = np.append(state, 1)
            state = np.append(state, 0)
        elif dy < 0:
            state = np.append(state, 1)
            state = np.append(state, 0)
        else:
            state = np.append(state, 0)
            state = np.append(state, 0)
    else:
        state = np.append(state, 0)
        state = np.append(state, 0)
        state = np.append(state, 0)
        state = np.append(state, 0)
    return state

def get_action(AI,state,epsilon,pred_showmode):
    #Epsilon : Probabilité que le Robot décide, et pas l'aléatoire (E [0,1])
    prediction = AI(torch.tensor(state).float())

    if pred_showmode:
        print(prediction)

    if random.uniform(0,1) > epsilon:
        action = random.randint(0,3)
        action_value = prediction[action]
    else:
        action = torch.argmax(prediction)
        action_value = torch.max(prediction)

    if pred_showmode:
        if action == 0:
            print(f"Action : {action} ==> Gauche")
        if action == 1:
            print(f"Action : {action} ==> Droite")
        if action == 2:
            print(f"Action : {action} ==> Haut")
        if action == 3:
            print(f"Action : {action} ==> Bas")
    return action,action_value

def ai_adjust(next_state,reward,AI,lossfun,optimizer,output,done,discount_factor):
    output_target = torch.tensor(reward).float()
    if not done:
        output_target += torch.max(AI(torch.tensor(next_state).float()))*discount_factor
    loss = lossfun(output,output_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return AI,lossfun,optimizer,loss.item()

def get_reward(done,has_eaten,snake,l,h,pred_showmode):
    if done and not len(snake.tab) >= l*h - snake.nbbloc:
        reward = -5.0
    elif done and len(snake.tab) == l*h - snake.nbbloc:
        reward = 1.0
    elif has_eaten:
        reward = 1.0
    else:
        reward = 0.0

    if pred_showmode:
        print(f"Reward : {reward}")
    return reward


def stronger_ai_adjust(state,reward,AI,lossfun,optimizer,output,done,discount_factor,snake,board,pred_showmode,depth,l,h):

    save_snake = copy.deepcopy(snake)
    save_board = copy.deepcopy(board)
    save_state = copy.deepcopy(state)

    output_target = torch.tensor(reward).float()

    for depthi in range(depth):

        if done:
            break

        action, action_value = get_action(AI, state, 1, pred_showmode)

        snake.orientate(action)
        board, done, has_eaten = snake.moove(board, l, h, done)

        if not done:
            state = get_stronger_state(snake,board,l,h)

        output_target += action_value*discount_factor*(depthi+1)

    loss = lossfun(output,output_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    snake = copy.deepcopy(save_snake)
    board = copy.deepcopy(save_board)
    state = copy.deepcopy(save_state)
    return AI,lossfun,optimizer,snake,board,state,loss.item()

def create_ai(learning_rate,importation,model_name):

    AI = nn.Sequential(

        nn.Linear(8, 16),
        nn.ReLU(),

        nn.Linear(16, 8),
        nn.ReLU(),

        nn.Linear(8, 4),

    )

    if importation and os.path.exists(model_name):
        AI.load_state_dict(torch.load(model_name))
        print("Importation effectuée")

    lossfun = nn.MSELoss()
    optimizer = optim.SGD(AI.parameters(),lr = learning_rate)

    return AI, lossfun, optimizer

def create_stronger_ai(l,h,learning_rate,importation,model_name):

    AI = nn.Sequential(

        nn.Linear(l*h*3+4, 32),
        nn.ReLU(),

        nn.Linear(32, 16),
        nn.ReLU(),

        nn.Linear(16, 4),

    )

    if importation and os.path.exists(model_name):
        AI.load_state_dict(torch.load(model_name))
        print("Importation effectuée")

    lossfun = nn.MSELoss()
    optimizer = optim.SGD(AI.parameters(),lr = learning_rate)

    return AI, lossfun, optimizer

def plot_resultats(ongoing_acc,ongoing_loss,plotsize,nb_buckets):

    acc = ongoing_acc[:plotsize]
    loss = ongoing_loss[:plotsize]

    acc_buckets = []
    loss_buckets = []

    bucket_size = plotsize//nb_buckets
    for bucket in range(nb_buckets-1):
        acc_buckets.append(np.mean(acc[bucket*bucket_size:(bucket+1)*bucket_size]))
        loss_buckets.append(np.mean(loss[bucket*bucket_size:(bucket+1)*bucket_size]))
    acc_buckets.append(np.mean(acc[nb_buckets-1 * bucket_size:]))
    loss_buckets.append(np.mean(loss[nb_buckets-1 * bucket_size:]))

    print(f"Final loss : {loss_buckets[-1]}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    ax[0].plot(loss_buckets,"bo")
    ax[0].set_ylabel("loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_title("Losses")

    ax[1].plot(acc_buckets,"ro")
    ax[1].set_ylabel("acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_title("Taille du serpent")

    plt.show()
