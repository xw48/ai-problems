#!/usr/bin/env python
# pikachu.py : this program is designed to recommend a move for Pikachu game.
# lee2074 & xw48, Feb. 28, 2017
#
# Problem: (how to formulate it)
# State space: this depends on the state, but this exponentially increase by increasing the depth. 
# For example, from initial state of 7x7 board, 14 w, 14 b.
# depth 1: 14
# depth 2: 41
# depth 3: 282
# depth 4: 4388
# depth 5: 5654
# depth 6: 80232
# depth 7: 121312
# Successor function: every possible moves including both pichu and pikachu.
# Edge weights: same edge weight, there is only heuristic value.
# How search algorithm works: 
# 1. deepen tree as deep as depth inputed.
# 2. evaluate the deepest nodes, and then compare scores
# 3. if w's turn choose a node which has max value, otherwise, b's turn, choose min value, and update alpha or beta value to biggest or smallest number according to whose turn. 
# 4. in a next step, go up to a depth, if there is another branch go down to a depth, and repeat again like 3, however, if alpha is bigger than beta it is no worth to evaluate.  
# 5. if all of branches are evaluated there are nodes which has been evaluated in previous step, and repeat 2-4.
# 6. eventually, adjacent state from initial state, in other words next possible moves, are found and have each values.
# 7. choose best value if w's turn the biggest one if b's turn the smallest one.
# Faced problems: successor function is very complicated to implement. 
# Assumtions: opponent has to choose his or her best way.


import sys
from heapq import heappush, heappop
import numpy as np
import gc
import ast
import copy
import time

timeout = 10.0
starttime = 0.0

def successor(b, color):
	result = []
	temp_b = copy.deepcopy(b)
	if color == 'w':
		o_color = 'b'
	else:
		o_color = 'w'
	
	for row in range(len(b)):
		for column in range(len(b[row])):
			
			#Successor for Pichu
			if b[row][column] == color:
				
				#move backward
				if row-1 in range(len(b)) and color == 'b':
					if b[row-1][column] == '.':
						temp_b[row][column] = '.'
						if color == 'b' and row-1 == 0:
							temp_b[row-1][column] = color.upper()
						else:
							temp_b[row-1][column] = color
						result.append(temp_b)
						temp_b = copy.deepcopy(b)
					#Jump and remove piece of opponent
					if row-2 in range(len(b)):	
						if b[row-1][column].lower() == o_color:
							if b[row-2][column] == '.':
								temp_b[row][column] = '.'
								temp_b[row-1][column] = '.'
								if color == 'b' and row-2 == 0:
									temp_b[row-2][column] = color.upper()
								else:
									temp_b[row-2][column] = color
								result.append(temp_b)
								temp_b = copy.deepcopy(b)
						
				#move forward
				if row+1 in range(len(b)) and color == 'w':
					if b[row+1][column] == '.':
						temp_b[row][column] = '.'
						if color == 'w' and row+1 == len(b)-1:
							temp_b[row+1][column] = color.upper()
						else:
							temp_b[row+1][column] = color
						result.append(temp_b)
						temp_b = copy.deepcopy(b)
					#Jump and remove piece of opponent
					if row+2 in range(len(b)):	
						if b[row+1][column].lower() == o_color:
							if b[row+2][column] == '.':
								temp_b[row][column] = '.'
								temp_b[row+1][column] = '.'
								if color == 'w' and row+2 == len(b)-1:
									temp_b[row+2][column] = color.upper()
								else:
									temp_b[row+2][column] = color
								temp_b[row+2][column] = color
								result.append(temp_b)
								temp_b = copy.deepcopy(b)
				
				#move left
				if column-1 in range(len(b[row])):
					if b[row][column-1] == '.':
						temp_b[row][column] = '.'
						temp_b[row][column-1] = color
						result.append(temp_b)
						temp_b = copy.deepcopy(b)
					#Jump and remove piece of opponent
					if column-2 in range(len(b)):	
						if b[row][column-1].lower() == o_color:
							if b[row][column-2] == '.':
								temp_b[row][column] = '.'
								temp_b[row][column-1] = '.'
								temp_b[row][column-2] = color
								result.append(temp_b)
								temp_b = copy.deepcopy(b)
														
				#move right
				if column+1 in range(len(b[row])):
					if b[row][column+1] == '.':
						temp_b[row][column] = '.'
						temp_b[row][column+1] = color
						result.append(temp_b)
						temp_b = copy.deepcopy(b)
					#Jump and remove piece of opponent
					if column+2 in range(len(b)):	
						if b[row][column+1].lower() == o_color:
							if b[row][column+2] == '.':
								temp_b[row][column] = '.'
								temp_b[row][column+1] = '.'
								temp_b[row][column+2] = color
								result.append(temp_b)
								temp_b = copy.deepcopy(b)

			#Successor for Pikachu
			elif b[row][column] == color.upper():
				#backward
				temp_b = copy.deepcopy(b)
				back_stop = len(b)
				back_ocolor = -1
				for i in range(1, len(b)):
					if row+i >= len(b):
						break
					if temp_b[row+i][column].lower() == o_color:
						if back_ocolor == -1:
							back_ocolor = row + i
						else:
							backward_stop = row + i
							break
					if temp_b[row+i][column].lower() == color:
						back_stop = row + i
						break
					if temp_b[row+i][column] == '.':
						continue

				for i in range(1, len(b)):
					if row+i >= back_stop:
						break
					if temp_b[row+i][column] == '.':
						temp_b[row][column] = '.'
						temp_b[row+i][column] = color.upper()
						if back_ocolor != -1 and back_ocolor < row+i:
							temp_b[back_ocolor][column] = '.'
						result.append(temp_b)
						temp_b = copy.deepcopy(b)

				#forward
				temp_b = copy.deepcopy(b)
				forward_stop = -1
				forward_ocolor = -1
				for i in range(1, len(b)):
					if row-i < 0:
						break
					if temp_b[row-i][column].lower() == o_color:
						if forward_ocolor == -1:
							forward_ocolor = row-i
						else:
							forward_stop = row-i
							break
					if temp_b[row-i][column].lower() == color:
						forward_stop = row-i
						break
					if temp_b[row-i][column] == '.':
						continue
				for i in range(1, len(b)):
					if row-i < 0 or row-i <= forward_stop:
						break
					if temp_b[row-i][column] == '.':
						temp_b[row][column] = '.'
						temp_b[row-i][column] = color.upper()
						if forward_ocolor != -1 and forward_ocolor > row-i:
							temp_b[forward_ocolor][column] = '.'
						result.append(temp_b)
						temp_b = copy.deepcopy(b)

				#move right
				temp_b = copy.deepcopy(b)
				right_stop = len(b)
				right_ocolor = -1
				for i in range(1, len(b)):
					if column+i >= len(b):
						break
					if temp_b[row][column+i].lower() == o_color:
						if right_ocolor == -1:
							right_ocolor = column + i
						else:
							right_stop = column+i
							break
					if temp_b[row][column+i].lower() == color:
						right_stop = column+i
						break
					if temp_b[row][column+i] == '.':
						continue
				for i in range(1, len(b)):
					if column+i >= right_stop:
						break
					if temp_b[row][column+i] == '.':
						temp_b[row][column] = '.'
						temp_b[row][column+i] = color.upper()
						if right_stop != -1 and right_ocolor < column+i:
							temp_b[row][right_ocolor] = '.'
						result.append(temp_b)
						temp_b = copy.deepcopy(b)

				#left
				temp_b = copy.deepcopy(b)
				left_stop = -1
				left_ocolor = -1
				for i in range(1, len(b)):
					if column-i < 0:
						break
					if temp_b[row][column-i].lower() == o_color:
						if left_ocolor == -1:
							left_ocolor = column-i
						else:
							left_stop = column-i
							break
					if temp_b[row][column-i].lower() == color:
						left_stop = column-i
						break
					if temp_b[row][column-i] == '.':
						continue
				for i in range(1, len(b)):
					if column-i < 0 or column-i <= left_stop:
						break
					if temp_b[row][column-i] == '.':
						temp_b[row][column] = '.'
						temp_b[row][column-i] = color.upper()
						if left_ocolor != -1 and left_ocolor >= column-i:
							temp_b[row][left_ocolor] = '.'
						result.append(temp_b)
						temp_b = copy.deepcopy(b)		
	return result


def remained_piece(b, color):
	return sum(b[row].count(color) for row in range(len(b))) + sum(b[row].count(color.upper()) for row in range(len(b)))


# 1. number of 'w's : 1
# 2. number of 'W's : 3
# 3. number of 'b's : -1
# 4. number of 'B's : -3
# 5. sum of w's distance to the end (being Pikachu) : -1/1000
# 6. sum of b's distance to the end (being Pikachu) : 1/1000
# x. number of w's possible moves : 1/10000
# x. number of b's possible moves : -1/10000

def heuristic(b):
	no_ws = sum(b[row].count('w') for row in range(len(b)))
	no_Ws = sum(b[row].count('W') for row in range(len(b)))*3
	no_bs = -sum(b[row].count('b') for row in range(len(b)))
	no_Bs = -sum(b[row].count('B') for row in range(len(b)))*3
	
	distance_w = 0.0
	for row in range(len(b)):
		for column in range(len(b[row])):
			if b[row][column] == 'w':
				distance_w += len(b)-1-row
	distance_w = -distance_w/1000
	
	distance_b = 0.0
	for row in range(len(b)):
		for column in range(len(b[row])):
			if b[row][column] == 'b':
				distance_b += row		
	distance_b = distance_b/1000
	
	#w_moves = len(successor(b, 'w'))/10000
	#b_moves = -len(successor(b, 'b'))/10000
	
	return no_ws + no_Ws + no_bs + no_Bs + distance_w + distance_b# + w_moves + b_moves 



def alphabeta(board_state, depth, alpha, beta, color):
	if depth == 0 or remained_piece(board_state, 'b') == 0 or remained_piece(board_state, 'w') == 0:
		return board_state
	if color == 'w':
		for next in successor(board_state, 'w'):
			temp = minmax(next, depth-1, alpha, beta, 'b')
			if temp >= alpha:
				alpha = temp
				bestmove = next
			if beta <= alpha:
				break
		return bestmove
	else:
		for next in successor(board_state, 'b'):
			temp = minmax(next, depth-1, alpha, beta, 'w')		
			if temp <= beta:
				beta = temp
				bestmove = next
			if beta >= alpha:
				break
		return bestmove
	
def minmax(board_state, depth, alpha, beta, color):
	global timeout
	global starttime

	if depth == 0 or remained_piece(board_state, 'b') == 0 or remained_piece(board_state, 'w') == 0 or (time.clock()-starttime) > timeout:
		return heuristic(board_state)

	if color == 'w':
		for next in successor(board_state, 'w'):
			temp = minmax(next, depth-1, alpha, beta, 'b')
			if temp >= alpha:
				alpha = temp
			if beta <= alpha:
				break
		return alpha
	else:
		for next in successor(board_state, 'b'):
			temp = minmax(next, depth-1, alpha, beta, 'w')		
			if temp <= beta:
				beta = temp
			if beta >= alpha:
				break
		return beta

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print 'usage:', sys.argv[0], 'n', 'w/b', 'state' 'timeout'
		sys.exit(-1)

	n = int(sys.argv[1])
	player = sys.argv[2]

	if player != 'w' and player != 'b':
		print 'bad current player format'
		sys.exit(-1)
	
	timeout = float(sys.argv[4])
	board_state = [[sys.argv[3][i*n+j] for j in range(n)]for i in range(n)]
	
	print 'Thinking! Please wait...'
	depth = 20

	starttime = time.clock()

	temp = alphabeta(board_state, depth, -100000, +100000, player)
	for row in range(0, n):
		for column in range(0, n):
			if temp[row][column] != board_state[row][column]:
				if temp[row][column] == player and board_state[row][column] == '.':
					next = (row+1, column+1, 'the Pichu')
				elif board_state[row][column] == player and temp[row][column] == '.':
					current = (row+1, column+1, 'the Pichu')
				elif temp[row][column] == player.upper() and board_state[row][column] == '.':
					next = (row+1, column+1, 'the Pikachu')
				elif board_state[row][column] == player.upper() and temp[row][column] == '.':
					current = (row+1, column+1, 'the Pikachu')
	
	print 'Hmm, I\'d recommend moving ' + current[2] + ' at row ' + str(current[0]) + ' column ' + str(current[1]) + ' to row ' + str(next[0]) + ' column ' + str(next[1]) + '.'
    #print 'New board:'
	result = ''
	for i in range(len(temp)):
		for j in range(len(temp[i])):
			result += temp[i][j]
	print result
	
		
