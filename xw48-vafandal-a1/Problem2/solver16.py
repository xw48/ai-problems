#!/usr/bin/env python
# solver16.py : this program is designed to solve 16-puzzle problem
# xw48, Vafandal, Feb. 7, 2017
# Problem: 16 puzzle is a variant of 15-puzzle, and there is no empty 
#          space on board. The only supported moves are L, R, U and D.
#
# State space: all arrangements of 1,..,16 on a 4X4 board
#
# Successor function: In 16 puzzle problem, all legal moves are L1,..,L4, R1,..R4, U1,..,U4, D1,..,D4. 
#					  Our successor function is s(board) = {Move(board), where Move belongs to {L1,..,L4, R1,..R4, U1,..,U4, D1,..,D4}
# edge weight: we do not define explicit weight for edges between board states. All edges are taken with edge 1.

# heuristic function: h(s) = Sum(Manhattan(i))/4. In other words, for each number i on board, we calculate the Manhattan distance 
#						to its right place, then sum them up, and divide the sum by 4. Because each move affects 4 tiles, h(s) <= h*(s) 
#						always holds. Only when all actions move their 4 tiles to right direction, h(s) = h*(s)
#
# Our program follows the definition of Astar algorithm. During implementation, we tried to use multiple heuristics and none of them 
# come up with an solution in reasonable time. To guarantee efficiency, we tried to maintain a list of visited/fringe states. Each 
# time we want to add a successor to fringe, we check this list. 

import sys
from heapq import heappush, heappop
import numpy as np
import gc
import ast

N = 4

# Count # of misplaced tiles
def heuristic(board_state):
	h = 0

	for i in range(0, N):
		for j in range(0, N):
			should_i = (board_state[i][j]-1)/N
			should_j = (board_state[i][j]-1)%N

			h += (min((should_i - i)%N, (i - should_i)%N) + min((should_j - j)%N, (j - should_j)%N))
	return float(h)/4.0

# check if is goal state
def is_goal_state(board_state):
	return np.array_equal(board_state, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# add moves to board
# action: L (left) R (right) U (up) D (down)
def add_move(board, action, num):
	if num not in range(1, 5):
		print 'Error movement'
		sys.exit(-1)

	if action == 'L':
		row = num-1
		return board[0:row] + [board[row][1:N] + [board[row][0],]] + board[row+1:]

	if action == 'R':
		row = num-1
		return board[0:row] + [[board[row][N-1],] + board[row][0:N-1]] + board[row+1:]

	if action == 'U':
		col = num-1
		nboard = [row[:] for row in board]
		for i in range(0, N):
			nboard[i][col] = board[(i+1)%N][col]
		return nboard

	if action == 'D':
		col = num-1
		nboard = [row[:] for row in board]
		for i in range(0, N):
			nboard[i][col] = board[(i-1)%N][col]
		return nboard

# get a list of successors
def successors(board, path):
	succs = []

	for action in ['L', 'R', 'U', 'D']:
		for num in range(1, 5):
			nboard = add_move(board, action, num)
			succs.append((nboard, path + [action+str(num),]))

	return succs

# solve 16 puzzle problem
def solve(initial_board):
	fringe = []
	initial_f = 0 + heuristic(initial_board)
	heappush(fringe, (initial_f, (initial_board, [])))

	in_fringe_boards = {}
	in_fringe_boards[str(initial_board)] = initial_f

	visited_boards = {}

	while len(fringe) > 0:
		item = heappop(fringe)
		item_f = item[0]
		item_board = item[1][0]
		item_path = item[1][1]

		print 'GET', item_f, item_board, item_path

		if str(item_board) in in_fringe_boards:
			del in_fringe_boards[str(item_board)]

		if is_goal_state(item_board):
			print ' '.join(str(x) for x in item_path)
			return True

		for s in successors(item_board, item_path):
			s_board = s[0]
			s_path = s[1]

			s_f = len(s_path) + heuristic(s_board)

			if (str(s_board) in visited_boards) and (s_f >= visited_boards[str(s_board)]):
				continue
			if (str(s_board) in in_fringe_boards) and (s_f >= in_fringe_boards[str(s_board)]):
				continue

			heappush(fringe, (s_f, (s_board, s_path)))
			in_fringe_boards[str(s_board)] = s_f

		visited_boards[str(item_board)] = item_f

	return False

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'usage:', sys.argv[0], '<input-board-filename>'
		sys.exit(-1)

	initial_board = []
	#read input board
	with open(sys.argv[1], 'r') as f:
		for l in f:
			initial_board.append([int(k) for k in l.split()])

	solve(initial_board)