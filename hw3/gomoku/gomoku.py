# -*- coding: utf-8 -*-
import numpy as np

def check_winner(L):
    N = len(L)
    if N < 5:
        return False
    else:
        s = np.sum(L[:5])
        if s == 5:
            return True
        if N > 5:
            for i in range(N-5):
                s = s - L[i] + L[i+5]
                if s == 5:
                    return True
        return False

class Board:
    def __init__(self, sz):
        self.sz = sz
        self.pbs = np.zeros((2, sz, sz), dtype=np.int)

    def add_move(self, p, x, y):
        self.pbs[p, x, y] = 1

        xd, xu = min(x, 4), min(self.sz-1-x, 4)
        yl, yr = min(y, 4), min(self.sz-1-y, 4)
        fs0, fs1 = min(xd, yl), min(xu, yr)
        bs0, bs1 = min(xu, yl), min(xd, yr)

        if check_winner(self.pbs[p, (x-xd):(x+xu+1), y]) or check_winner(self.pbs[p, x, (y-yl):(y+yr+1)]):
            return True
        elif check_winner(self.pbs[p, np.arange((x-fs0), (x+fs1+1)), np.arange((y-fs0), (y+fs1+1))]):
            return True
        elif check_winner(self.pbs[p, np.arange((x+bs0), (x-bs1-1), -1), np.arange((y-bs0), (y+bs1+1))]):
            return True
        else:
            return False


class Gomoku:
    def __init__(self, board_sz=11, gui=False):
        self.board_sz = board_sz
        self.board = Board(board_sz)
        self.number = np.zeros((board_sz, board_sz), dtype=int)
        self.k = 1  # step number
        self.result = 0
        if gui:
            self.gui = GameGUI(board_sz)
        else:
            self.gui = None


    def reset(self):
        self.board.pbs.fill(0)
        self.number.fill(0)
        self.k = 1
        self.result = 0

    def copy(self):  # copy the game, not the UI
        g = Gomoku(self.board_sz)
        g.board.pbs = np.copy(self.board.pbs)
        g.number = np.copy(self.number)
        g.k = self.k
        g.result = self.result
        return g

    def draw(self):
        print(self.board.pbs[0, :, :]-self.board.pbs[1, :, :])
        if self.gui:
            self.gui._draw_background()
            self.gui._draw_chessman(self.board.pbs[0, :, :]-self.board.pbs[1, :, :], self.number)

    # execute a move
    def execute_move(self, p, x, y):
        assert np.sum(self.board.pbs[:, x, y]) == 0

        win = self.board.add_move(p, x, y)
        self.number[x][y] = self.k
        self.k += 1
        return win

    # main loop
    def play(self, p1, p2):
        players = {0:p1, 1:p2}
        pi = 0
        self.draw()
        while True:
            x, y = players[pi].get_move(self.board.pbs)
            if x < 0:
                break  
            win = self.execute_move(pi, x, y)
            self.draw()
            
            if win:
                self.result = 1-2*pi
                break
            if np.sum(self.board.pbs) == self.board_sz*self.board_sz:
                break
            
            pi = (pi+1) % 2



class RandomPlayer:
    def __init__(self, id):
        self.id = id

    def get_move(self, board):
        b = (board[0, :, :] + board[1, :, :]) - 1
        ix, jx = np.nonzero(b)
        idx = [i for i in zip(ix, jx)]
        return idx[np.random.choice(len(idx))]


if __name__ == "__main__":
    
    import sys
    if len(sys.argv) > 1:
    # A user plays game with a random player
        from gamegui import GameGUI, GUIPlayer  # do not import gamegui if you don't have pygame or not on local machine.
        g = Gomoku(11, True)

        p1 = RandomPlayer(0)
        p2 = GUIPlayer(1, g.gui)
        print('start GUI game, close window to exit.')    
        g.play(p1, p2)

        g.gui.draw_result(g.result)
        g.gui.wait_to_exit()

    else:
    # Two random player play 100 rounds of non-GUI game
        g = Gomoku()
        p1 = RandomPlayer(0)
        p2 = RandomPlayer(1)
        for i in range(100):
            g.play(p1, p2)
            print(i, g.result)
            g.reset()


    


