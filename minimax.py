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
class QAgent_Minimax:
    def __init__(self, mark, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.mark = mark
        self.q_table = defaultdict(lambda: np.zeros(9))
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
        return max(valid_q, key=lambda x: x[1])[0]
    
    def minimax_future(self, next_state, next_available, opponent):
        # Compute minimax assuming opponent plays optimally

        board = list(next_state)
        winner = check_winner(board)
        if winner:
            # terminal evaluation
            if winner == self.mark:
                return 1
            elif winner == opponent.mark:
                return -1
            else:
                return 0

        if not next_available:
            return 0

        values = []
        for my_move in next_available:
            new_board = list(next_state)
            new_board[my_move] = self.mark
            opp_avail = available_moves(new_board)

            winner = check_winner(new_board)
            if winner:
                if winner == self.mark:
                    values.append(1)
                elif winner == opponent.mark:
                    values.append(-1)
                else:
                    values.append(0)
                continue

            if not opp_avail:
                values.append(0)
                continue

            # Opponent minimizes outcome
            opp_values = []
            for opp_move in opp_avail:
                opp_board = list(new_board)
                opp_board[opp_move] = opponent.mark
                opp_winner = check_winner(opp_board)
                if opp_winner == self.mark:
                    opp_values.append(1)
                elif opp_winner == opponent.mark:
                    opp_values.append(-1)
                elif opp_winner == "Draw":
                    opp_values.append(0)
                else:
                    # Fall back to q-values if not terminal
                    opp_values.append(opponent.q_table[tuple(opp_board)][opp_move])
            values.append(min(opp_values))

        return max(values) # agent 1 maximizes worst-case outcome
    
    def update(self, state, action, reward, next_state, next_available, opponent):
        if action is None:
            return
        old_value = self.q_table[state][action]

        if reward != 0 or not next_available:
            future = 0
        else:
            future = self.minimax_future(next_state, next_available, opponent)

        new_value = old_value + self.lr * (reward + self.gamma * future - old_value)
        self.q_table[state][action] = new_value

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---- Training via self-play ---- #
def train(episodes = 300000):
    agent_X = QAgent_Minimax("X")
    agent_O = QAgent_Minimax("O")

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
                agent_X.update(prev_X[0], prev_X[1], 0, state, available, opponent=agent_O)
            elif current.mark == "O" and prev_O is not None:
                agent_O.update(prev_O[0], prev_O[1], 0, state, available, opponent=agent_X)

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
                    agent_X.update(prev_X[0], prev_X[1], reward_X, next_state, [], opponent=agent_O)
                if prev_O is not None:
                    agent_O.update(prev_O[0], prev_O[1], reward_O, next_state, [], opponent=agent_X)

                # Update current move
                if current.mark == "X":
                    agent_X.update(state, action, reward_X, next_state, [], opponent=agent_O)
                else:
                    agent_O.update(state, action, reward_O, next_state, [], opponent=agent_X)
                break

            # Save previous move
            if current.mark == "X":
                prev_X = (state, action)
            else:
                prev_O = (state, action)

            state = next_state
            current, other = other, current

    return agent_X, agent_O


# ---- Evaluate performance ---- #
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
    with open("qtables/agent_X_qtable_minimax.pkl", "wb") as f:
        pickle.dump(dict(agent_X.q_table), f)

    with open("qtables/agent_O_qtable_minimax.pkl", "wb") as f:
        pickle.dump(dict(agent_O.q_table), f)