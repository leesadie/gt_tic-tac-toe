# to run: streamlit run streamlit_app.py

import streamlit as st
import time
from collections import defaultdict
import numpy as np
import random
import pickle
from q_nash import QAgent
from minimax import QAgent_Minimax
from maximax import QAgent_Maximax

# ---- Tic-tac-toe environment ---- #
def create_board():
    return [" "] * 9

def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == " "]

def check_winner(board):
    win_patterns = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for a,b,c in win_patterns:
        if board[a]==board[b]==board[c]!=" ":
            return board[a]
    if " " not in board:
        return "Draw"
    return None

def board_to_state(board):
    return tuple(board)


# ---- Agents ---- #
def load_agents(strategy):
    if strategy == "nash":
        agent_X = QAgent("X")
        agent_O = QAgent("O")

        # Load Q-tables
        with open("qtables/agent_X_qtable_nash.pkl", "rb") as f:
            q_X = pickle.load(f)
        agent_X.q_table = defaultdict(lambda: np.zeros(9), q_X)

        with open("qtables/agent_O_qtable_nash.pkl", "rb") as f:
            q_O = pickle.load(f)
        agent_O.q_table = defaultdict(lambda: np.zeros(9), q_O)

    elif strategy == "minimax":
        agent_X = QAgent_Minimax("X")
        agent_O = QAgent_Minimax("O")

        # Load Q-tables
        with open("qtables/agent_X_qtable_minimax.pkl", "rb") as f:
            q_X = pickle.load(f)
        agent_X.q_table = defaultdict(lambda: np.zeros(9), q_X)

        with open("qtables/agent_O_qtable_minimax.pkl", "rb") as f:
            q_O = pickle.load(f)
        agent_O.q_table = defaultdict(lambda: np.zeros(9), q_O)

    elif strategy == "maximax":
        agent_X = QAgent_Maximax("X")
        agent_O = QAgent_Maximax("O")

        # Load Q-tables
        with open("qtables/agent_X_qtable_maximax.pkl", "rb") as f:
            q_X = pickle.load(f)
        agent_X.q_table = defaultdict(lambda: np.zeros(9), q_X)

        with open("qtables/agent_O_qtable_maximax.pkl", "rb") as f:
            q_O = pickle.load(f)
        agent_O.q_table = defaultdict(lambda: np.zeros(9), q_O)

    return agent_X, agent_O


# ---- UI config ---- #
st.set_page_config(page_title="2-Agent Tic-Tac-Toe", layout="wide")

st.markdown("""
<style>
    :root {
        --board-margin-top: 200px;   
        --board-cell-size: 140px;
    }
    /* Page/title */
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    h1 {text-align: center; margin-bottom: 4px;}
            
    h2 {
        text-align: center; 
        font-size: 20px; font-weight: normal;
        padding: 0rem 18rem;      
    }
    
    /* Strategy label above tabs */
    .strategy-section {text-align: left; margin-bottom: -10px; margin-top: 20px;}
            
    /* Result banner overlay */
    .result-banner {
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -300%);
        font-size: 36px;
        font-weight: bold;
        background-color: rgba(219, 219, 219, 0.95);
        padding: 12px 40px;
        border: 2px #B5B5B5;
        border-radius: 5px;
        color: rgba(0, 0, 0);      
    }
            
    /* Board wrapper */
    .board-container {
        position: relative;
        display: flex; 
        justify-content: center;
        margin-top: var(--board-margin-top);
        width: 540px;
        margin-left: auto;
        margin-right: auto;
    }
    .board-row {
        display: flex;
        justify-content: center;        
    }
    .board-cell {
        width: 140px; height: 140px;
        font-size: 72px; font-weight: bold;
        display: flex; align-items: center; justify-content: center;
        border: 8px solid #ABBECF;        
    }
    .board-cell.top {border-top: none;}
    .board-cell.left {border-left: none;}
    .board-cell.right {border-right: none;}
    .board-cell.bottom {border-bottom: none;}
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1>Game-Theoretic Q-Learning for Tic-Tac-Toe</h1>", unsafe_allow_html=True)
st.markdown('<h2>Agents have been trained with different strategies using Q-learning and are loaded to visualize strategies in new games. Find code <a href="https://github.com/leesadie/gt_tic-tac-toe" target="_blank">here</a>.</h2>', unsafe_allow_html=True)

# Strategy selection tabs (placeholder)
st.markdown('<div class="strategy-section">Select strategy:</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Nash equilibrium", "Minimax", "Maximax"])

# Render board upon page load
def render_board(board):
    for i in range(0, 9, 3):
        row_html = '<div class="board-row">'
        for j in range(3):
            idx = i + j
            cell = board[idx]
            if cell == "X":
                mark = '<span style="color:#2069B2">X</span>'
            elif cell == "O":
                mark = '<span style="color:#174075">O</span>'
            else:
                mark = ""

            classes = ["board-cell"]
            if i == 0: classes.append("top")
            if i == 6: classes.append("bottom")
            if j == 0: classes.append("left")
            if j == 2: classes.append("right")
            row_html += f'<div class="{" ".join(classes)}">{mark}</div>'
        row_html += "</div>"
        st.markdown(row_html, unsafe_allow_html=True)

# ---- Nash equilibrium ---- #
with tab1:
    # Info
    st.info("A change in one player's strategy has no effect if the other player's strategies remain fixed; expected outcome is a draw if both players play optimally.")
    
    # Load agents
    agent_X, agent_O = load_agents(strategy="nash")

    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("### Agent X")
    with col3:
        st.markdown("### Agent O")

    # Placeholder for the board
    with col2:
        board_container = st.container()
        with board_container:
            board_placeholder = st.empty()
            result_placeholder = st.empty()

    # Show empty grid upon load
    with board_placeholder.container():
        render_board(create_board())

    col_left, col_center, col_right = st.columns([2,0.5,2])
    with col_center:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        play = st.button("Play new game", key="play_button_nash")

    if play:
        board = create_board()
        state = board_to_state(board)
        current, other = agent_X, agent_O

        # Game loop
        while True:
            available = available_moves(board)
            if not available:
                result = "Draw"
                break

            action = current.choose_action(state, available, greedy=True)
            board[action] = current.mark

            # Render board as a 3x3 grid
            with board_placeholder.container():
                render_board(board)  

            time.sleep(1.0)

            winner = check_winner(board)
            if winner:
                result = winner
                break

            state = board_to_state(board)
            current, other = other, current

        # Show result
        if result == "Draw":
            result_text = "Draw!"
        else:
            result_text = f"{result} wins!"
        result_placeholder.markdown(
            f'<div class="result-banner">{result_text}</div>', unsafe_allow_html=True
        )


# ---- Minimax ---- #
with tab2:
    # Info
    st.info("Assuming the opponent always plays optimally, each player maximizes their worst-case outcome; expected outcome is a draw if both players play optimally.")
    
    # Load agents
    agent_X, agent_O = load_agents(strategy="minimax")

    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("### Agent X")
    with col3:
        st.markdown("### Agent O")

    # Placeholder for the board
    with col2:
        board_container = st.container()
        with board_container:
            board_placeholder = st.empty()
            result_placeholder = st.empty()

    # Show empty grid upon load
    with board_placeholder.container():
        render_board(create_board())

    col_left, col_center, col_right = st.columns([2,0.5,2])
    with col_center:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        play = st.button("Play new game", key="play_button_minimax")

    if play:
        board = create_board()
        state = board_to_state(board)
        current, other = agent_X, agent_O

        # Game loop
        while True:
            available = available_moves(board)
            if not available:
                result = "Draw"
                break

            action = current.choose_action(state, available, greedy=True)
            board[action] = current.mark

            # Render board as a 3x3 grid
            with board_placeholder.container():
                render_board(board)  

            time.sleep(1.0)

            winner = check_winner(board)
            if winner:
                result = winner
                break

            state = board_to_state(board)
            current, other = other, current

        # Show result
        if result == "Draw":
            result_text = "Draw!"
        else:
            result_text = f"{result} wins!"
        result_placeholder.markdown(
            f'<div class="result-banner">{result_text}</div>', unsafe_allow_html=True
        )


# ---- Maximax ---- #
with tab3:
    # Info
    st.info("Assuming the opponent plays to maximize the player's payoff, each player chooses the best-case outcome; expected outcome is a win for the first mover since the first mover's optimism dominates.")
    
    # Load agents
    agent_X, agent_O = load_agents(strategy="maximax")

    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("### Agent X")
    with col3:
        st.markdown("### Agent O")

    # Placeholder for the board
    with col2:
        board_container = st.container()
        with board_container:
            board_placeholder = st.empty()
            result_placeholder = st.empty()

    # Show empty grid upon load
    with board_placeholder.container():
        render_board(create_board())

    col_left, col_center, col_right = st.columns([2,0.5,2])
    with col_center:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        play = st.button("Play new game", key="play_button_maximax")

    if play:
        board = create_board()
        state = board_to_state(board)
        
        # Random alternation - so sometimes O starts
        if random.random() < 0.5:
            current, other = agent_X, agent_O
        else:
            current, other = agent_O, agent_X

        # Game loop
        while True:
            available = available_moves(board)
            if not available:
                result = "Draw"
                break

            action = current.choose_action(state, available, greedy=True)
            board[action] = current.mark

            # Render board as a 3x3 grid
            with board_placeholder.container():
                render_board(board)  

            time.sleep(1.0)

            winner = check_winner(board)
            if winner:
                result = winner
                break

            state = board_to_state(board)
            current, other = other, current

        # Show result
        if result == "Draw":
            result_text = "Draw!"
        else:
            result_text = f"{result} wins!"
        result_placeholder.markdown(
            f'<div class="result-banner">{result_text}</div>', unsafe_allow_html=True
        )