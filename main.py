#!/usr/bin/env python3
import sys
import random
import numpy as np

class Settings(object):
    def __init__(self):
        self.timebank = None 
        self.time_per_move = None
        self.player_names = None 
        self.your_bot = None
        self.your_botid = None
        self.field_width = None
        self.field_height = None


class Field(object):
    def __init__(self):
        self.field_state = None
        self.prev_state = None

    def update_field(self, celltypes, settings):
        self.prev_state == copy.deepcopy(self.field_state)
        self.field_state = [[] for _ in range(settings.field_height)]
        n_cols = settings.field_width
        for idx, cell in enumerate(celltypes):
            row_idx = idx // n_cols
            self.field_state[row_idx].append(cell)


class State(object):
    def __init__(self):
        self.settings = Settings()
        self.field = Field()
        self.round = 0
        self.prev = None #Records the previous move of the bot
        self.o_prev = None #Records the oponents previous move


def parse_communication(text):
    """ Return the first word of the communication - that's the command """
    return text.strip().split()[0] 


def settings(text, state):
    """ Handle communication intended to update game settings """
    tokens = text.strip().split()[1:] # Ignore token 0, it's the string "settings".
    cmd = tokens[0]
    if cmd in ('timebank', 'time_per_move', 'your_botid', 'field_height', 'field_width'):
        # Handle setting integer settings.
        setattr(state.settings, cmd, int(tokens[1]))
    elif cmd in ('your_bot',):
        # Handle setting string settings.
        setattr(state.settings, cmd, tokens[1])
    elif cmd in ('player_names',):
        # Handle setting lists of strings.
        setattr(state.settings, cmd, tokens[1:])
    else:
        raise NotImplementedError('Settings command "{}" not recognized'.format(text))
    

def update(text, state):
    """ Handle communication intended to update the game """
    tokens = text.strip().split()[2:] # Ignore tokens 0 and 1, those are "update" and "game" respectively.
    cmd = tokens[0]
    if cmd in ('round',):
        # Handle setting integer settings.
        setattr(state.settings, 'round', int(tokens[1]))
    if cmd in ('field',):
        # Handle setting the game board.
        celltypes = tokens[1].split(',')
        state.field.update_field(celltypes, state.settings)


def action(text, state):
    """ Handle communication intended to prompt the bot to take an action """
    tokens = text.strip().split()[1:] # Ignore token 0, it's the string "action".
    cmd = tokens[0]
    if cmd in ('move',):
        return make_move(state)
    else:
        raise NotImplementedError('Action command "{}" not recognized'.format(text))

def is_on_edge(target):
    if len(target) == 1 and (target == 0 or target == 6):
        #Assumed the number sent was a column
        return true
    elif target[0] == 0 or target[0] == 5:
        return True
    elif target[1] == 0 or target[1] == 6:
        return True
    return False

def oponent_move(prev_state, field_state):
    #TODO: Check what move the opponent has made.
    pass

def check_pattern(cell, ID, row_shft, col_shft,
                  pattern ={'01': 0, '0-1': 0, '10': 0, '-10': 0,
                       '11': 0, '-1-1': 0, '1-1': 0, '-11': 0}):
    
    """ Checks for any patern a newly made move has formed. """
    #Recursive function.
    #pattern holds the number of linked tokens in the 8 possible
    #directions that can form a winning pattern.
    
    row, col = cell[0], cell[1]
    for r in row_shft:
        for c in col_shft:
            #The second part of the condition is to not match the given cell itself
            if state.field.field_state[row+r][col+c]==ID and not(r == 0 and c==0 ): 
                pattern[str(r)+str(c)] += 1 #Increment the pattern number.
                                            #The direction being probed in is represented
                                            #by combining the shift in row and column respectively.
                check_pattern([row+r, col+c], ID, [r], [c], pattern) #Further probing in the direction the matching cell was found
    return pattern

def make_move(state):

    # TODO: Implement bot logic here    
    if not state.round:
        move = 3
        state.prev = move
        return 'place_disc {}'.format(move)
    
    #p = check_pattern(state.prev, state.settings.your_botid, [0,1,-1], [0,1,-1])
    #print (p)
    if getattr(state.field, 'field_state')[0][state.prev] == '.': #Checks if the column played previously is not full
        move = state.prev
        
    else:
        if state.prev == 6:
            move = 0
        move = state.prev +1    #If full plays the next column

def main():
    command_lookup = { 'settings': settings, 'update': update, 'action': action }
    state = State()
    input(">    ") #sys.stdin reads whatever is stored in input.
    
    for input_msg in sys.stdin:
        cmd_type = parse_communication(input_msg)
        command = command_lookup[cmd_type]

        # Call the correct command. 
        res = command(input_msg, state)

        # Assume if the command generates a string as output, that we need 
        # to "respond" by printing it to stdout.
        if isinstance(res, str):
            print(res)
            sys.stdout.flush()
        


if __name__ == '__main__':
    main()
