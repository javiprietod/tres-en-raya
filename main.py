class Board:
    def __init__(self):
        self.state = '0'*9

    def __str__(self):
        XO = {0: '   ', 1: ' X ', 2: ' O '}
        str = '╔' + ('═'*3 + '╦')*2 + '═'*3 + '╗'
        for idx, elem in enumerate(self.state):
            if idx % 3 == 0:
                str += '\n║'
            str += XO[int(elem)] + '║'
            if idx % 3 == 2 and idx != 8:
                str += '\n╠' + ('═'*3 + '╬')*2 + '═'*3 + '╣'
        str += '\n╚' + ('═'*3 + '╩')*2 + '═'*3 + '╝'
        return str


class Agente:
    def __init__(self, board):
        self.board = board
        self.player = 2
        