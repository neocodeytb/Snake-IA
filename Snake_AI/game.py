import copy

import numpy as np
import pygame as pg
from ai import *
from src import *

class Snake:
    def __init__(self,l,h,blocs_mode,start_lenght):
        self.x = random.randint(0,l-1)
        self.y = random.randint(0,h-1)
        self.orientation = 1
        self.tab = np.array([[self.x - i,self.y] for i in range(start_lenght)])
        self.tick = 0
        self.nbbloc = 0
        self.blocs_mode = blocs_mode
        self.coord_apple = [0,0]
        self.tab_blocs = []
        #1 : droite, 2 : bas, -1 : gauche, -2 : haut

    def moove(self,board,l,h,done):
        has_eaten = False
        if self.orientation % 2 == 0:
            self.y += self.orientation // 2
        else:
            self.x += self.orientation
        self.tick += 1
        done = self.detection_mort(board, l, h, done)
        longueur = len(self.tab)
        if not done:
            board, done,has_eaten = self.eat_apple(board, done)
            if self.tab[longueur-1][0] >= 0 and self.tab[longueur-1][0] < l and self.tab[longueur-1][1] >= 0 and self.tab[longueur-1][1] < h:
                board[self.tab[longueur-1][0],self.tab[longueur-1][1]] = 0
            board[self.x,self.y] = 1
        for idx in range(longueur-1):
            self.tab[longueur - idx - 1] = self.tab[longueur - idx - 2]
        self.tab[0] = [self.x,self.y]
        if has_eaten:
            board, done = self.spawn_apple(board, done)
            if self.blocs_mode:
                board,done = self.spawn_bloc(board,done)
        board[self.coord_apple[0],self.coord_apple[1]] = 2
        return board,done,has_eaten

    def orientate(self,action):
        if action == 0 and not self.orientation == 1:
            self.orientation = -1 #Gauche
        if action == 1 and not self.orientation == -1:
            self.orientation = 1 #Droite
        if action == 2 and not self.orientation == 2:
            self.orientation = -2 #Haut
        if action == 3 and not self.orientation == -2:
            self.orientation = 2 #Bas

    def detection_mort(self,board,l,h,done):
        if self.x < 0 or self.x >= l or self.y < 0 or self.y >= h or board[self.x,self.y] == 1 or board[self.x,self.y] == 3 or self.tick > len(self.tab)+l*h:
            done = True
        return done

    def eat_apple(self,board,done):
        has_eaten = False
        if board[self.x,self.y] == 2:
            has_eaten = True
            self.tick = 0
            if self.orientation % 2 == 0:
                nouvelle_case = np.array([self.tab[len(self.tab) - 1][0],self.tab[len(self.tab) - 1][1] - self.orientation//2])
            else:
                nouvelle_case = np.array([self.tab[len(self.tab) - 1][0] - self.orientation,self.tab[len(self.tab) - 1][1]])
            self.tab = np.append(self.tab,[nouvelle_case],axis = 0)
        return board,done,has_eaten

    def spawn_bloc(self,board,done):
        spawn_possibles = np.where(np.logical_and(board != 1, board != 2, board != 3))
        if len(spawn_possibles[0]) == 0:
            done = True
        else:
            spawn = np.random.choice(len(spawn_possibles[0]))
            board[spawn_possibles[0][spawn], spawn_possibles[1][spawn]] = 3
            self.nbbloc += 1
            self.tab_blocs.append([spawn_possibles[0][spawn], spawn_possibles[1][spawn]])
        return board, done

    def spawn_apple(self,board, done):
        spawn_possibles = np.where(board == 0)
        if len(spawn_possibles[0]) == 0:
            done = True
        else:
            spawn = np.random.choice(len(spawn_possibles[0]))
            board[spawn_possibles[0][spawn], spawn_possibles[1][spawn]] = 2
            self.coord_apple = [spawn_possibles[0][spawn], spawn_possibles[1][spawn]]
        return board, done

    def display(self,screen,sources,t):
        if self.orientation == -1:
            screen.blit(sources["head_left"],(self.x*t,self.y*t))
        if self.orientation == 1:
            screen.blit(sources["head_right"],(self.x*t,self.y*t))
        if self.orientation == -2:
            screen.blit(sources["head_top"],(self.x*t,self.y*t))
        if self.orientation == 2:
            screen.blit(sources["head_bottom"],(self.x*t,self.y*t))
        longueur = len(self.tab)

        for idx in range(1,longueur - 1):
            if self.tab[idx][0] == self.tab[idx+1][0] and  self.tab[idx][0] == self.tab[idx-1][0]:
                screen.blit(sources["vertical_tail"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][1] == self.tab[idx+1][1] and  self.tab[idx][1] == self.tab[idx-1][1]:
                screen.blit(sources["horizontal_tail"], (self.tab[idx][0] * t, self.tab[idx][1] * t))

            elif self.tab[idx][0] + 1 == self.tab[idx+1][0] and  self.tab[idx][1] - 1 == self.tab[idx-1][1]:
                screen.blit(sources["top_to_left"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][0] + 1 == self.tab[idx+1][0] and  self.tab[idx][1] + 1 == self.tab[idx-1][1]:
                screen.blit(sources["bottom_to_left"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][0] - 1 == self.tab[idx + 1][0] and self.tab[idx][1] - 1 == self.tab[idx - 1][1]:
                screen.blit(sources["right_to_top"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][0] - 1 == self.tab[idx + 1][0] and self.tab[idx][1] + 1 == self.tab[idx - 1][1]:
                screen.blit(sources["left_to_bottom"], (self.tab[idx][0] * t, self.tab[idx][1] * t))

            elif self.tab[idx][0] + 1 == self.tab[idx-1][0] and  self.tab[idx][1] - 1 == self.tab[idx+1][1]:
                screen.blit(sources["top_to_left"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][0] + 1 == self.tab[idx-1][0] and  self.tab[idx][1] + 1 == self.tab[idx+1][1]:
                screen.blit(sources["bottom_to_left"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][0] - 1 == self.tab[idx - 1][0] and self.tab[idx][1] - 1 == self.tab[idx + 1][1]:
                screen.blit(sources["right_to_top"], (self.tab[idx][0] * t, self.tab[idx][1] * t))
            elif self.tab[idx][0] - 1 == self.tab[idx - 1][0] and self.tab[idx][1] + 1 == self.tab[idx + 1][1]:
                screen.blit(sources["left_to_bottom"], (self.tab[idx][0] * t, self.tab[idx][1] * t))

        if self.tab[longueur - 2][0] > self.tab[longueur - 1][0]:
            screen.blit(sources["tail_left"],(self.tab[longueur - 1][0] * t, self.tab[longueur - 1][1] * t))
        elif self.tab[longueur - 2][0] < self.tab[longueur - 1][0]:
            screen.blit(sources["tail_right"],(self.tab[longueur - 1][0] * t, self.tab[longueur - 1][1] * t))
        if self.tab[longueur - 2][1] > self.tab[longueur - 1][1]:
            screen.blit(sources["tail_top"],(self.tab[longueur - 1][0] * t, self.tab[longueur - 1][1] * t))
        elif self.tab[longueur - 2][1] < self.tab[longueur - 1][1]:
            screen.blit(sources["tail_bottom"],(self.tab[longueur - 1][0] * t, self.tab[longueur - 1][1] * t))

def global_init(l,h,t,fps,numepochs,display,random_epsilon,epsilon_value,blocs_mode,pred_showmode,strenght,discount_factor,start_lenght):
    pg.init()
    pg.font.init()
    screen = pg.display.set_mode((l*t,h*t))
    clock = pg.time.Clock()
    ongoing_acc = np.empty(numepochs)
    ongoing_loss = np.empty(numepochs)
    font = pg.font.Font(None, 30)
    best_game_memory = np.empty(0,dtype = object)
    return screen,clock,l,h,t,fps,numepochs,display,ongoing_acc,False,0,random_epsilon,epsilon_value,blocs_mode,pred_showmode,strenght,discount_factor,font,best_game_memory,ongoing_loss,start_lenght

def init_game(l,h,epsilon,rdm_epsilon,blocs_mode,start_lenght):
    board = np.zeros((l,h),dtype=int)
    snake = Snake(l,h,blocs_mode,start_lenght)
    for element in snake.tab:
        if element[0] >= 0 and element[1] >= 0:
            board[element[0],element[1]] = 1
    board,done = snake.spawn_apple(board,False)
    if rdm_epsilon:
        epsilon = random.uniform(0.0,1)
    return board,snake,done,epsilon

def get_state(snake,board,l,h):

    state = np.zeros(8)

    if snake.x + 1 < l and snake.y >= 0 and snake.y < h:
        if board[snake.x + 1,snake.y] == 1 or board[snake.x + 1,snake.y] == 3:
            state[0] = 1
    elif snake.x + 1 >= l:
        state[0] = 1

    if snake.y + 1 < h and snake.x >= 0 and snake.x < l:
        if board[snake.x, snake.y + 1] == 1 or board[snake.x, snake.y + 1] == 3:
            state[2] = 1
    elif snake.y + 1 >= h:
        state[2] = 1

    if snake.x - 1 >= 0 and snake.y >= 0 and snake.y < h:
        if board[snake.x - 1,snake.y] == 1 or board[snake.x - 1,snake.y] == 3:
            state[1] = 1
    elif snake.x - 1 < 0:
        state[1] = 1

    if snake.y - 1 >= 0 and snake.x >= 0 and snake.x < l:
        if board[snake.x, snake.y - 1] == 1 or board[snake.x, snake.y - 1] == 3:
            state[3] = 1
    elif snake.y - 1 < 0:
        state[3] = 1


    food = np.where(board == 2)
    if food[0].size == 1:

        food_x = food[0][0]
        food_y = food[1][0]

        dx = food_x - snake.x
        dy = food_y - snake.y

        if dx > 0:
            state[4] = 1
        elif dx < 0:
            state[5] = 1

        if dy > 0:
            state[6] = 1
        elif dy < 0:
            state[7] = 1

    return state

def get_reward(done,has_eaten,snake,l,h,pred_showmode):
    if done and not len(snake.tab) >= l*h - snake.nbbloc:
        reward = -2.0
    elif done and len(snake.tab) == l*h - snake.nbbloc:
        reward = 1.0
    elif has_eaten:
        reward = 1.0
    else:
        reward = 0.0

    if pred_showmode:
        print(f"Reward : {reward}")
    return reward

def display_state(state,l,pred_showmode):
    if pred_showmode:
        display_tab = []
        for idx in range(len(state) // 3):
            if state[idx*3] == 1:
                display_tab.append("q")
            elif state[idx*3 + 1] == 1:
                display_tab.append("p")
            elif state[idx*3+2] == 1:
                display_tab.append("s")
            else:
                display_tab.append("o")
        idx = 0
        for element in display_tab:
            print(element,end="")
            idx += 1
            if idx == l:
                print("")
                idx = 0

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

def env_step(action,board,snake,l,h,done,pred_showmode):
    state = get_state(snake,board,l,h)
    snake.orientate(action)
    board,done,has_eaten = snake.moove(board,l,h,done)
    next_state = get_state(snake,board,l,h)
    display_state(state, l, pred_showmode)
    reward = get_reward(done,has_eaten,snake,l,h,pred_showmode)
    return state,next_state,reward,done,board,snake

def env_stronger_step(action,board,snake,l,h,done,pred_showmode,next_state):
    snake.orientate(action)
    board, done, has_eaten = snake.moove(board, l, h, done)
    reward = get_reward(done, has_eaten, snake, l, h, pred_showmode)
    if not done:
        next_state = get_stronger_state(snake, board, l, h)
        #display_state(next_state, l, pred_showmode) (Affichage de l'état pour débug / analyse)
    return next_state, reward, done, board, snake

def game(screen,clock,board,snake,l,h,t,fps,done,display_mode,AI,lossfun,optimizer,epsilon,stop,rdm_epsilon,sources,pred_showmode,discount_factor,epochi,font):
    next_state = get_state(snake,board,l,h)
    game_memory = np.empty(0,dtype = object)
    total_loss,iteration = 0,0
    while not done:
        game_memory = np.append(game_memory,copy.deepcopy(snake))
        action,action_value = get_action(AI,next_state,epsilon,pred_showmode)
        state,next_state,reward,done,board,snake = env_step(action,board,snake,l,h,done,pred_showmode)
        AI, lossfun, optimizer,lossi = ai_adjust(next_state,reward,AI,lossfun,optimizer,action_value,done,discount_factor)
        display(screen,board,l,h,t,display_mode,clock,fps,sources,snake,epochi,font)
        display_mode,done,stop,epsilon,rdm_epsilon,fps,pred_showmode,discount_factor = ihm(display_mode,done,screen,AI,stop,epsilon,rdm_epsilon,fps,pred_showmode,discount_factor)
        total_loss+=lossi
        iteration += 1
        if len(snake.tab) >= l*h - snake.nbbloc:
            game_memory = np.append(game_memory, copy.deepcopy(snake))
            done = True
    ongoing_loss = total_loss/iteration
    return len(snake.tab),stop,display_mode,epsilon,rdm_epsilon,AI,lossfun,optimizer,fps,pred_showmode,discount_factor,game_memory,ongoing_loss

def stronger_game(screen,clock,board,snake,l,h,t,fps,done,display_mode,AI,lossfun,optimizer,epsilon,stop,rdm_epsilon,sources,pred_showmode,depth,discount_factor,epochi,font,start_lenght):
    state = get_stronger_state(snake,board,l,h)
    game_memory = np.empty(0,dtype = object)
    total_loss,iteration = 0,0
    while not done:
        game_memory = np.append(game_memory, copy.deepcopy(snake))
        action,action_value = get_action(AI,state,epsilon,pred_showmode)
        state,reward,done,board,snake = env_stronger_step(action,board,snake,l,h,done,pred_showmode,state)
        AI, lossfun, optimizer,snake,board,state,loss = stronger_ai_adjust(state,reward,AI,lossfun,optimizer,action_value,done,discount_factor,snake,board,pred_showmode,depth,l,h)
        display(screen, board, l, h, t, display_mode, clock, fps, sources, snake,epochi,font)
        display_mode, done, stop, epsilon, rdm_epsilon, fps, pred_showmode,depth,discount_factor,start_lenght = stronger_ihm(display_mode, done, screen, AI, stop,epsilon, rdm_epsilon, fps,pred_showmode,depth,discount_factor,start_lenght,l,h)
        total_loss += loss
        iteration += 1
        if len(snake.tab) >= l*h - snake.nbbloc:
            game_memory = np.append(game_memory, copy.deepcopy(snake))
            done = True
    game_loss = total_loss/iteration
    return len(snake.tab), stop, display_mode, epsilon, rdm_epsilon, AI, lossfun, optimizer, fps, pred_showmode,depth,discount_factor,game_memory,game_loss,start_lenght


def display(screen,board,l,h,t,display_mode,clock,fps,sources,snake,epochi,font):
    if display_mode:
        screen.fill((0,0,0))
        for i in range(l):
            for j in range(h):
                pg.draw.rect(screen,(0 + 50*((i + j) % 2),0 + 50*((i + j) % 2),0 + 50*((i + j) % 2)),(i*t,j*t,t,t))
                if board[i,j] == 2:
                    screen.blit(sources["apple"],(i*t,j*t))
                if board[i,j] == 3:
                    screen.blit(sources["wall"],(i*t,j*t))
        snake.display(screen,sources,t)
        text_score = font.render(f"Iteration : {epochi+1}", True, (255, 255, 0))
        screen.blit(text_score, (10, 10, 80, 80))
        pg.display.flip()
        clock.tick(fps)
def ihm(display_mode,done,screen,AI,stop,epsilon,rdm_epsilon,fps,pred_showmode,discount_factor):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            done,stop = True,True
            torch.save(AI.state_dict(),"stronger_ai_6x6.pth")
            print("Sauvegarde effectuée")
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                display_mode = not display_mode
                screen.fill((0, 0, 0))
                pg.display.flip()
            if event.key == pg.K_UP:
                epsilon = min(1,epsilon + 0.05)
                print(f"Valeur de Epsilon : {epsilon}")
            if event.key == pg.K_DOWN:
                epsilon = max(0,epsilon - 0.05)
                print(f"Valeur de Epsilon : {epsilon}")
            if event.key == pg.K_a:
                rdm_epsilon = not rdm_epsilon
                print(f"Epsilon aléatoire : {rdm_epsilon}")
            if event.key == pg.K_z:
                epsilon = 0
                print(f"Epsilon fixé à {epsilon}, aléatoire maximum")
            if event.key == pg.K_e:
                epsilon = 1
                print(f"Epsilon fixé à {epsilon}, le robot joue")
            if event.key == pg.K_q:
                fps = max(0,fps - 1)
                print(f"FPS : {fps}")
            if event.key == pg.K_d:
                fps = min(100,fps + 1)
                print(f"FPS : {fps}")
            if event.key == pg.K_s:
                pred_showmode = not pred_showmode
                print(f"Affichage des prédictions : {pred_showmode}")
            if event.key == pg.K_c:
                discount_factor = max(0,discount_factor-0.05)
                print(f"Discount Factor : {discount_factor}")
            if event.key == pg.K_v:
                discount_factor = min(1,discount_factor+0.05)
                print(f"Discount Factor : {discount_factor}")

    return display_mode,done,stop,epsilon,rdm_epsilon,fps,pred_showmode,discount_factor

def stronger_ihm(display_mode,done,screen,AI,stop,epsilon,rdm_epsilon,fps,pred_showmode,depth,discount_factor,start_lenght,l,h):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            done,stop = True,True
            torch.save(AI.state_dict(),"stronger_ai_6x6.pth")
            print("Sauvegarde effectuée")
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                display_mode = not display_mode
                screen.fill((0, 0, 0))
                pg.display.flip()
            if event.key == pg.K_UP:
                epsilon = min(1,epsilon + 0.05)
                print(f"Valeur de Epsilon : {epsilon}")
            if event.key == pg.K_DOWN:
                epsilon = max(0,epsilon - 0.05)
                print(f"Valeur de Epsilon : {epsilon}")
            if event.key == pg.K_a:
                rdm_epsilon = not rdm_epsilon
                print(f"Epsilon aléatoire : {rdm_epsilon}")
            if event.key == pg.K_z:
                epsilon = 0
                print(f"Epsilon fixé à {epsilon}, aléatoire maximum")
            if event.key == pg.K_e:
                epsilon = 1
                print(f"Epsilon fixé à {epsilon}, le robot joue")
            if event.key == pg.K_q:
                fps = max(0,fps - 1)
                print(f"FPS : {fps}")
            if event.key == pg.K_d:
                fps = min(100,fps + 1)
                print(f"FPS : {fps}")
            if event.key == pg.K_s:
                pred_showmode = not pred_showmode
                print(f"Affichage des prédictions : {pred_showmode}")
            if event.key == pg.K_w:
                depth = max(0,depth - 1)
                print(f"Profondeur : {depth}")
            if event.key == pg.K_x:
                depth = min(50,depth + 1)
                print(f"Profondeur : {depth}")
            if event.key == pg.K_c:
                discount_factor = max(0,discount_factor-0.05)
                print(f"Discount Factor : {discount_factor}")
            if event.key == pg.K_v:
                discount_factor = min(1,discount_factor+0.05)
                print(f"Discount Factor : {discount_factor}")
            if event.key == pg.K_m:
                start_lenght = max(4,start_lenght-1)
                print(f"Taille de départ du serpent : {start_lenght}")
            if event.key == pg.K_p:
                start_lenght = min(l*h-1,start_lenght+1)
                print(f"Taille de départ du serpent : {start_lenght}")
    return display_mode,done,stop,epsilon,rdm_epsilon,fps,pred_showmode,depth,discount_factor,start_lenght

def play_best_game(best_game_memory,score_max,l,h,t,fps,sources):
    print(f"Score Max : {score_max}")
    pg.init()
    screen = pg.display.set_mode((l * t, h * t))
    clock = pg.time.Clock()
    for snake in best_game_memory:
        screen.fill((0, 0, 0))
        for i in range(l):
            for j in range(h):
                pg.draw.rect(screen, (0 + 50 * ((i + j) % 2), 0 + 50 * ((i + j) % 2), 0 + 50 * ((i + j) % 2)),
                             (i * t, j * t, t, t))
                if i == snake.coord_apple[0] and j == snake.coord_apple[1]:
                    screen.blit(sources["apple"], (i * t, j * t))
        for bloc in snake.tab_blocs:
            screen.blit(sources["wall"],(bloc[0]*t,bloc[1]*t))
        snake.display(screen, sources, t)
        pg.display.flip()
        clock.tick(fps)
    pg.quit()
