class Board:
    def __init__(self):
        self.state = '0'*9
    
    def __str__(self):
        XO = {0: ' ', 1: 'X', 2: 'O'}
        str = '-------'
        for i in range(3):
            str += '|' + self.state[i*3:i*3+3] + '|'
            for j in range(3):
                str += self.state[i*3+j]
            str += '-------'
        str += '-------'
        return str

class Agente:
    def __init__(self, board):
        self.board = board
        self.player = 2
    