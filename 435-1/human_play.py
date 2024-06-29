# -*- coding: utf-8 -*-
"""
human VS AI models with GUI using Tkinter

@author: yzq_GZHU
"""

from __future__ import print_function
import pickle
import tkinter as tk
from tkinter import messagebox
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy


class Human:
    def __init__(self, player=None):
        self.player = player
        self.move = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        while self.move is None:
            board.root.update()
        move = self.move
        self.move = None
        return move

    def __str__(self):
        return "Human {}".format(self.player)


class GomokuGUI:
    def __init__(self, root, board, human, ai):
        self.root = root
        self.root.title("Gomoku")
        self.board = board
        self.board.init_board()
        self.human = human
        self.ai = ai
        self.human_move = None
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click)
        self.draw_board()
        self.play_game()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(8):
            self.canvas.create_line(i * 50, 0, i * 50, 400)
            self.canvas.create_line(0, i * 50, 400, i * 50)

    def draw_piece(self, x, y, player):
        if player == 1:
            self.canvas.create_oval(x * 50 + 10, y * 50 + 10, x * 50 + 40, y * 50 + 40, fill="white")
        else:
            self.canvas.create_oval(x * 50 + 10, y * 50 + 10, x * 50 + 40, y * 50 + 40, fill="black")

    def click(self, event):
        x, y = event.x // 50, event.y // 50
        move = self.board.location_to_move([y, x])
        if move != -1 and move in self.board.availables:
            self.human.move = move

    def play_game(self):
        self.human.set_player_ind(1)
        self.ai.set_player_ind(2)
        while True:
            self.root.update()
            if self.board.current_player == 1:
                move = self.human.get_action(self.board)
                self.board.do_move(move)
                x, y = self.board.move_to_location(move)
                self.draw_piece(y, x, self.board.current_player)
            else:
                move = self.ai.get_action(self.board)
                self.board.do_move(move)
                x, y = self.board.move_to_location(move)
                self.draw_piece(y, x, self.board.current_player)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    winner = "Black" if winner == 1 else "White"
                    messagebox.showinfo("Game Over", f"{winner} wins!")
                else:
                    messagebox.showinfo("Game Over", "It's a tie!")
                self.board.init_board()
                self.draw_board()
                break


def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model2'
    try:
        root = tk.Tk()
        board = Board(root=root, width=width, height=height, n_in_row=n)

        # Load the trained policy_value_net
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        human = Human()

        gomoku = GomokuGUI(root, board, human, mcts_player)
        root.mainloop()
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
