import copy

from game import *
from src import create_sources

def parametric_experiment(depth,epsilon,discount_factor,epochi):

    #Fonction pour automatiser les paramêtres d'apprentissage si l'on est AFK
    return depth,epsilon,discount_factor

screen,clock,l,h,t,fps,num_epochs,display_mode,ongoing_acc,stop,plotsize,rdm_epsilon,epsilon,blocs_mode,pred_showmode,strenght,discount_factor,font,best_game_memory,ongoing_loss,start_lenght = global_init(l=6,h=6,t=100,fps = 10,numepochs = 10000000,display = True,
                                                                                                                      random_epsilon = False,epsilon_value = 1,blocs_mode = True,pred_showmode = False,strenght = False,discount_factor=0.1,start_lenght = 4)
sources = create_sources(t)

if not strenght:
    score_max = 0
    AI,lossfun,optimizer = create_ai(learning_rate=0.01,importation = True,model_name = "ai.pth")
    for epochi in range(num_epochs):
        board, snake, done, epsilon = init_game(l, h, epsilon, rdm_epsilon, blocs_mode,start_lenght)
        ongoing_acc[epochi], stop, display_mode, epsilon, rdm_epsilon, AI, lossfun, optimizer, fps, pred_showmode,discount_factor,game_memory,ongoing_loss[epochi] = game(screen,clock,board,snake,l, h,t, fps,done,display_mode, AI,lossfun,optimizer,epsilon,stop,rdm_epsilon,sources,pred_showmode,discount_factor,epochi,font)
        print(f"Score Game {epochi + 1} : {ongoing_acc[epochi]}")

        if ongoing_acc[epochi] > score_max:
            best_game_memory = copy.deepcopy(game_memory)
            score_max = ongoing_acc[epochi]

        if stop:
            plotsize = epochi
            plot_resultats(ongoing_acc,ongoing_loss, plotsize,nb_buckets=100)
            play_best_game(best_game_memory,score_max,l,h,t,fps,sources)
            break

        if score_max >= l*h - snake.nbbloc:
            print("Le serpent a fini le jeu")

else:
    AI,lossfun,optimizer = create_stronger_ai(l,h,learning_rate=0.01,importation = False,model_name = "stronger_ai_6x6.pth")
    depth = 1  #Profondeur de recherche, a augmenter graduellement et doucement avec le temps /// E [1;+oo]

    #Augmenter la profondeur, le discount factor au fur et à mesure, alterner entre exploration-exploitation et finir l'entrainement sur une longue phase d'exploitation,
    #et la commencer par une longue phase d'exploration

    score_max = 0
    for epochi in range(num_epochs):
        board, snake, done, epsilon = init_game(l, h, epsilon, rdm_epsilon, blocs_mode,start_lenght)
        depth, epsilon, discount_factor = parametric_experiment(depth,epsilon,discount_factor,epochi)
        ongoing_acc[epochi], stop, display_mode, epsilon, rdm_epsilon, AI, lossfun, optimizer, fps, pred_showmode,depth,discount_factor,game_memory,ongoing_loss[epochi],start_lenght = stronger_game(screen,clock,board,snake,l, h,t, fps,done,display_mode, AI,lossfun,optimizer,epsilon,stop,rdm_epsilon,sources,pred_showmode,depth,discount_factor,epochi,font,start_lenght)
        print(f"Score Game {epochi + 1} : {ongoing_acc[epochi]}")

        if ongoing_acc[epochi] > score_max:
            best_game_memory = copy.deepcopy(game_memory)
            score_max = ongoing_acc[epochi]

        if stop:
            plotsize = epochi
            plot_resultats(ongoing_acc,ongoing_loss, plotsize,nb_buckets=100)
            play_best_game(best_game_memory,score_max,l,h,t,fps,sources)
            break

        if score_max >= l*h - snake.nbbloc:
            print("Le serpent a fini le jeu")



