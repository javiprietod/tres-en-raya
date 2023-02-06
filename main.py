class Board:
    def __init__(self):
        self.state = '0'*9
    
    def __str__(self):
        str = '-------'
        for i in range(3):
            str += '|' + self.state[i*3:i*3+3] + '|'
            for j in range(3):
                str += self.state[i*3+j]
            str += '-------'
        str += '-------'
        return str

