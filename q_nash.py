import random
from collections import defaultdict
import numpy as np
import pickle

# ---- Tic-tac-toe environment ---- #
def create_board():
    return [" "] * 9

def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == " "]

def check_winner(board):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for a, b, c in win_patterns:
        if board[a] == board[b] == board[c] != " ":
            return board[a]
    if " " not in board:
        return "Draw"
    return None

# Make board hashable for Q-table keys
def board_to_state(board):
    return tuple(board)

# ---- Q-learning agent ---- #
class QAgent:
    def __init__(self, mark, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.mark = mark
        self.q_table = defaultdict(lambda: np.zeros(9))  # 9 action slots
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state, available, greedy=False):
        if not available:
            return None
        if (not greedy) and (random.random() < self.epsilon):
            return random.choice(available)
        q_values = self.q_table[state]
        valid_q = [(a, q_values[a]) for a in available]
        return max(valid_q, key=lambda x: x[1])[0] # best valid move

    def update(self, state, action, reward, next_state, next_available):
        if action is None:
            return
        old_value = self.q_table[state][action]
        future = 0 if not next_available else max(self.q_table[next_state][a] for a in next_available)
        new_value = old_value + self.lr * (reward + self.gamma * future - old_value)
        self.q_table[state][action] = new_value
        # decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ---- Training via self-play ---- #
def train(episodes=300000):
    agent_X = QAgent("X")
    agent_O = QAgent("O")

    for ep in range(episodes):
        board = create_board()
        state = board_to_state(board)

        # Alternate which agent starts
        if ep % 2 == 0:
            current, other = agent_X, agent_O
        else:
            current, other = agent_O, agent_X

        # Track previous moves for proper Q-updates
        prev_X, prev_O = None, None

        while True:
            available = available_moves(board)
            if not available:
                break

            action = current.choose_action(state, available)
            board[action] = current.mark
            next_state = board_to_state(board)
            next_available = available_moves(board)
            winner = check_winner(board)

            # Update previous moves with zero reward
            if current.mark == "X" and prev_X is not None:
                agent_X.update(prev_X[0], prev_X[1], 0, state, available)
            elif current.mark == "O" and prev_O is not None:
                agent_O.update(prev_O[0], prev_O[1], 0, state, available)

            if winner:
                # Assign final rewards
                if winner == "Draw":
                    reward_X, reward_O = 0, 0
                elif winner == "X":
                    reward_X, reward_O = 1, -1
                else:
                    reward_X, reward_O = -1, 1
                
                # Update last moves
                if prev_X is not None:
                    agent_X.update(prev_X[0], prev_X[1], reward_X, next_state, [])
                if prev_O is not None:
                    agent_O.update(prev_O[0], prev_O[1], reward_O, next_state, [])

                # Update current move
                if current.mark == "X":
                    agent_X.update(state, action, reward_X, next_state, [])
                else:
                    agent_O.update(state, action, reward_O, next_state, [])
                break

            # Save previous move
            if current.mark == "X":
                prev_X = (state, action)
            else:
                prev_O = (state, action)

            state = next_state
            current, other = other, current

    return agent_X, agent_O

# ---- Evaluate approximate Nash equilibrium ---- #
def evaluate(agent_X, agent_O, games=5000):
    results = {"X": 0, "O": 0, "Draw": 0}
    for g in range(games):
        board = create_board()
        state = board_to_state(board)
        current, other = (agent_X, agent_O) if g % 2 == 0 else (agent_O, agent_X)

        while True:
            available = available_moves(board)
            action = current.choose_action(state, available, greedy=True)
            board[action] = current.mark
            winner = check_winner(board)
            state = board_to_state(board)

            if winner:
                results[winner] += 1
                break
            current, other = other, current
    return results

# ---- Visualize one match post-training ---- #
def play_match(agent_X, agent_O):
    board = create_board()
    state = board_to_state(board)
    current, other = agent_X, agent_O

    print("New game\n")
    while True:
        available = available_moves(board)
        if not available:
            print("Game over: Draw\n")
            break
        action = current.choose_action(state, available, greedy=True)
        board[action] = current.mark

        # Print board
        for i in range(0, 9, 3):
            print(" | ".join(board[i:i+3]))
        print("-----")

        winner = check_winner(board)
        if winner:
            print(f"Game over: {winner}\n")
            break
        
        state = board_to_state(board)
        current, other = other, current

# ---- Run training and evaluation ---- #
if __name__ == "__main__":
    agent_X, agent_O = train(episodes=300000)
    print("Exploration rates after training:")
    print("Agent X epsilon:", agent_X.epsilon)
    print("Agent O epsilon:", agent_O.epsilon)

    results = evaluate(agent_X, agent_O, games=5000)
    print("Evaluation results:", results)

    play_match(agent_X, agent_O)

    # Save Q-tables
    with open("agent_X_qtable.pkl", "wb") as f:
        pickle.dump(dict(agent_X.q_table), f)

    with open("agent_O_qtable.pkl", "wb") as f:
        pickle.dump(dict(agent_O.q_table), f)