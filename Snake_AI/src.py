import pygame as pg



def create_sources(t):
    sources = {}
    for name in ("apple","bottom_to_left","head_bottom","head_left","head_right","head_top","horizontal_tail","vertical_tail","left_to_bottom","right_to_top","tail_bottom","tail_left","tail_right","tail_top","top_to_left","vertical_tail","wall"):
        sources[name] = pg.transform.scale(pg.image.load(name + ".png"),(t,t))
    return sources