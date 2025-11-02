import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import time
import streamlit as st

# ========== Core RL + Environment ==========

# Board utilities
EMPTY = " "
PLAYERS = ("X", "O")

Coord = Tuple[int, int]  # (row, col)
Action = int             # 0..8 index
Board = Tuple[str, ...]  # 9-tuple of "X","O"," "
State = Tuple[str, str]  # (board_str, player_to_move)

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diags
]

# --- Board transform helpers for symmetry augmentation ---
def index_to_coord(i: int) -> Tuple[int, int]:
    return divmod(i, 3)

def coord_to_index(r: int, c: int) -> int:
    return 3 * r + c

def transform_coord(r: int, c: int, name: str) -> Tuple[int, int]:
    # name in {'identity','rot90','rot180','rot270','flip_h','flip_v','diag','anti'}
    if name == 'identity':
        return r, c
    if name == 'rot90':
        return c, 2 - r
    if name == 'rot180':
        return 2 - r, 2 - c
    if name == 'rot270':
        return 2 - c, r
    if name == 'flip_h':
        return r, 2 - c
    if name == 'flip_v':
        return 2 - r, c
    if name == 'diag':
        return c, r
    if name == 'anti':
        return 2 - c, 2 - r
    raise ValueError(f"Unknown transform {name}")

TRANSFORMS = ['identity', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'diag', 'anti']

def transform_board(board: Board, name: str) -> Board:
    arr = [' '] * 9
    for i, v in enumerate(board):
        r, c = index_to_coord(i)
        r2, c2 = transform_coord(r, c, name)
        arr[coord_to_index(r2, c2)] = v
    return tuple(arr)

def transform_action(a: int, name: str) -> int:
    r, c = index_to_coord(a)
    r2, c2 = transform_coord(r, c, name)
    return coord_to_index(r2, c2)


def board_to_str(board: Board) -> str:
    return "".join(board)

def str_to_board(s: str) -> Board:
    return tuple(s)

def empty_board() -> Board:
    return tuple([EMPTY]*9)

def available_actions(board: Board) -> List[Action]:
    return [i for i, v in enumerate(board) if v == EMPTY]

def winner(board: Board) -> Optional[str]:
    for a,b,c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None

def terminal(board: Board) -> bool:
    return winner(board) is not None or EMPTY not in board

def next_player(p: str) -> str:
    return "O" if p == "X" else "X"

def apply_action(board: Board, player: str, action: Action) -> Board:
    assert board[action] == EMPTY
    b = list(board)
    b[action] = player
    return tuple(b)

# Q-table: Q[(board_str, player)][action] = value
QTable = Dict[str, Dict[str, float]]  # JSON-friendly dict

def state_key(board: Board, player: str) -> str:
    return f"{board_to_str(board)}|{player}"

def get_q_for_state(Q: QTable, board: Board, player: str) -> Dict[int, float]:
    key = state_key(board, player)
    if key not in Q:
        Q[key] = {}  # initialize lazily
    # Only return legal actions; implicit zeros otherwise
    legal = available_actions(board)
    q = {a: Q[key].get(str(a), 0.0) for a in legal}
    return q

def set_q_value(Q: QTable, board: Board, player: str, action: int, value: float):
    key = state_key(board, player)
    if key not in Q:
        Q[key] = {}
    Q[key][str(action)] = float(value)

def epsilon_greedy_action(Q: QTable, board: Board, player: str, epsilon: float, rng: np.random.Generator) -> int:
    legal = available_actions(board)
    if not legal:
        return -1
    if rng.random() < epsilon:
        return int(rng.choice(legal))
    q = get_q_for_state(Q, board, player)
    # Break ties deterministically but evenly (sort by action index)
    best_val = max(q.get(a, 0.0) for a in legal)
    best_actions = [a for a in legal if q.get(a, 0.0) == best_val]
    return int(sorted(best_actions)[0])

@dataclass
class QLearningConfig:
    alpha: float = 0.5         # learning rate
    gamma: float = 0.99        # discount
    epsilon: float = 1.0       # exploration during training

def terminal_reward(final_board: Board, for_player: str) -> float:
    w = winner(final_board)
    if w is None:
        return 0.0  # draw
    return 1.0 if w == for_player else -1.0

def q_update(Q: QTable, s_board: Board, s_player: str, a: int, r: float, s2_board: Board, s2_player: str, cfg: QLearningConfig):
    # Q(s,a) += alpha * [ r + gamma*max_a' Q(s', a') - Q(s,a) ]
    q_s = get_q_for_state(Q, s_board, s_player)
    old = q_s.get(a, 0.0)
    if terminal(s2_board):
        target = r
    else:
        q_s2 = get_q_for_state(Q, s2_board, s2_player)
        target = r + cfg.gamma * (max(q_s2.values()) if q_s2 else 0.0)
    new_val = old + cfg.alpha * (target - old)
    set_q_value(Q, s_board, s_player, a, new_val)

def self_play_train(Q: QTable, episodes: int, cfg: QLearningConfig, seed: Optional[int] = None, progress_callback=None, progress_interval: Optional[int] = None, epsilon_min: Optional[float] = None, alpha_min: Optional[float] = None, symmetry: bool = True):
    """
    Train by self-play. Optional progress_callback(i, total) will be called periodically.
    progress_interval: call callback every progress_interval episodes (defaults to max(1, episodes//100)).
    """
    rng = np.random.default_rng(seed)
    if progress_interval is None and progress_callback is not None:
        progress_interval = max(1, episodes // 100)

    for i in range(episodes):
        # Linear schedules for alpha and epsilon
        frac = i / max(1, episodes - 1)
        eps_start = float(cfg.epsilon)
        a_start = float(cfg.alpha)
        eps_min = float(epsilon_min) if epsilon_min is not None else eps_start
        a_min = float(alpha_min) if alpha_min is not None else a_start
        current_epsilon = max(eps_min, eps_start + (eps_min - eps_start) * frac)
        current_alpha = max(a_min, a_start + (a_min - a_start) * frac)
        # Per-episode cfg to use for q_update
        cfg_ep = QLearningConfig(alpha=current_alpha, gamma=cfg.gamma, epsilon=current_epsilon)
        board = empty_board()
        player = "X" if rng.random() < 0.5 else "O"
        # Play until terminal; update after each move from the acting player's perspective
        while not terminal(board):
            a = epsilon_greedy_action(Q, board, player, current_epsilon, rng)
            if a == -1:
                break
            next_board = apply_action(board, player, a)
            if terminal(next_board):
                r = terminal_reward(next_board, player)
                # Update for original and symmetric variants
                if symmetry:
                    for t in TRANSFORMS:
                        s_t = transform_board(board, t)
                        s2_t = transform_board(next_board, t)
                        a_t = transform_action(a, t)
                        q_update(Q, s_t, player, a_t, r, s2_t, next_player(player), cfg_ep)
                else:
                    q_update(Q, board, player, a, r, next_board, next_player(player), cfg_ep)
                break
            # intermediate reward is 0; next state belongs to the opponent
            if symmetry:
                for t in TRANSFORMS:
                    s_t = transform_board(board, t)
                    s2_t = transform_board(next_board, t)
                    a_t = transform_action(a, t)
                    q_update(Q, s_t, player, a_t, 0.0, s2_t, next_player(player), cfg_ep)
            else:
                q_update(Q, board, player, a, 0.0, next_board, next_player(player), cfg_ep)
            board = next_board
            player = next_player(player)

        if progress_callback is not None and (progress_interval is None or (i % progress_interval == 0)):
            try:
                progress_callback(i + 1, episodes)
            except Exception:
                # ignore progress callback failures
                pass

# ========== Streamlit UI State & Helpers ==========

def init_state():
    ss = st.session_state
    ss.setdefault("Q", {})  # Q table
    ss.setdefault("board", empty_board())
    ss.setdefault("current_player", "X")
    ss.setdefault("mode", "Human vs AI")  # or "2 Humans"
    ss.setdefault("human_plays_as", "X")
    ss.setdefault("learn_during_play", False)
    ss.setdefault("alpha", 0.5)
    ss.setdefault("gamma", 0.99)
    ss.setdefault("epsilon_play", 1.0)   # AI exploration when playing
    ss.setdefault("epsilon_train", 1.0)  # AI exploration when training
    ss.setdefault("episodes", 500)
    ss.setdefault("rng_seed", 42)
    # Persist a random Generator so moves are not re-seeded on every call
    ss.setdefault("rng", None)
    if ss["rng"] is None:
        try:
            ss["rng"] = np.random.default_rng(int(ss.get("rng_seed", None)))
        except Exception:
            ss["rng"] = np.random.default_rng()
    ss.setdefault("random_start", False)
    ss.setdefault("auto_play", False)
    ss.setdefault("auto_play_delay", 2)
    ss.setdefault("ai_auto_start_time", None)
    ss.setdefault("last_rng_seed", None)
    ss.setdefault("preview_counter", 0)
    ss.setdefault("preview_action", None)
    ss.setdefault("last_ai_action", None)
    ss.setdefault("show_explain", True)  # show Q explanation when AI to move

def reset_board(randomize_first: bool = False):
    st.session_state.board = empty_board()
    if randomize_first:
        # Use persistent RNG if available
        rng = st.session_state.get("rng") or np.random.default_rng()
        st.session_state.current_player = "X" if rng.random() < 0.5 else "O"
    else:
        st.session_state.current_player = "X"

def reset_q():
    st.session_state.Q = {}

def save_q_json() -> str:
    return json.dumps(st.session_state.Q)

def load_q_json(text: str):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            st.session_state.Q = obj
            st.success("Loaded Q-table.")
        else:
            st.error("Invalid Q-table format.")
    except Exception as e:
        st.error(f"Failed to load Q-table: {e}")

# ========== Gameplay Handlers ==========

def maybe_ai_move():
    """If it's AI's turn in Human vs AI, make a move (with epsilon_play)."""
    mode = st.session_state.mode
    if mode != "Human vs AI":
        return
    board = st.session_state.board
    if terminal(board):
        return

    human_as = st.session_state.human_plays_as
    ai_as = next_player(human_as)

    if st.session_state.current_player == ai_as:
        # Let AI move
        cfg = QLearningConfig(
            alpha=float(st.session_state.alpha),
            gamma=float(st.session_state.gamma),
            epsilon=float(st.session_state.epsilon_play)
        )
        # Use a persistent RNG from session_state so different moves are possible
        rng = st.session_state.get("rng") or np.random.default_rng()
        a = epsilon_greedy_action(st.session_state.Q, board, ai_as, cfg.epsilon, rng)
        if a != -1:
            s = (board, ai_as)
            next_b = apply_action(board, ai_as, a)
            # Optional learn while you play (on-policy update with immediate reward 0 unless terminal)
            if st.session_state.learn_during_play:
                r = terminal_reward(next_b, ai_as) if terminal(next_b) else 0.0
                q_update(st.session_state.Q, s[0], s[1], a, r, next_b, next_player(ai_as), cfg)
            st.session_state.board = next_b
            st.session_state.current_player = next_player(ai_as)

            # Record which action the AI actually took for highlighting in the explanation
            st.session_state["last_ai_action"] = int(a)
            # Clear any preview action (preview is separate and should be set explicitly)
            st.session_state["preview_action"] = None

            # After AI moves, reset any auto-play timer
            st.session_state["ai_auto_start_time"] = None

def handle_cell_click(idx: int):
    board = st.session_state.board
    if terminal(board) or board[idx] != EMPTY:
        return

    mode = st.session_state.mode
    cur = st.session_state.current_player

    # If Human vs AI and it's AI's turn, ignore click
    if mode == "Human vs AI":
        human_as = st.session_state.human_plays_as
        if cur != human_as:
            return

    # Human move
    st.session_state.board = apply_action(board, cur, idx)
    st.session_state.current_player = next_player(cur)
    # Do not trigger an immediate AI response here so the UI can display
    # the AI's Q-values (explanation) before the AI acts. The UI will
    # render an explicit "AI: Make move" button when it's the AI's turn.

    # If auto-play is enabled and it's now the AI's turn, start the auto-play timer
    try:
        mode = st.session_state.mode
        human_as = st.session_state.human_plays_as
        ai_as = next_player(human_as)
        if mode == "Human vs AI" and st.session_state.current_player == ai_as and st.session_state.get("auto_play", False):
            st.session_state["ai_auto_start_time"] = time.time()
    except Exception:
        pass

# ========== UI Components ==========

def render_controls():
    with st.sidebar:
        st.header("Settings")
        st.selectbox("Mode", ["Human vs AI", "2 Humans"], key="mode")
        if st.session_state.mode == "Human vs AI":
            st.radio("Human plays as", ["X", "O"], key="human_plays_as", horizontal=True)
            st.checkbox("Learn while you play (online updates)", key="learn_during_play")
            st.slider("AI exploration while playing (ε)", 0.0, 1.0, key="epsilon_play", step=0.05)

        st.divider()
        st.subheader("Randomness / Seed")
        # Expose rng seed so instructors can reproduce runs
        # RNG seed input; automatically reseed when seed value changes
        st.number_input("RNG seed (integer)", key="rng_seed", value=int(st.session_state.get("rng_seed", 42)), step=1, format="%d")
        # Auto-reseed when the seed value has changed since last render
        try:
            curr_seed = int(st.session_state.get("rng_seed", 42))
            last = st.session_state.get("last_rng_seed")
            if last is None or curr_seed != last:
                st.session_state["rng"] = np.random.default_rng(curr_seed)
                st.session_state["last_rng_seed"] = curr_seed
                st.info(f"RNG automatically reseeded with {curr_seed}")
        except Exception:
            st.error("Invalid RNG seed")
        if st.button("Reseed RNG"):
            try:
                st.session_state["rng"] = np.random.default_rng(int(st.session_state.rng_seed))
                st.session_state["last_rng_seed"] = int(st.session_state.rng_seed)
                st.success(f"RNG reseeded with {st.session_state.rng_seed}")
            except Exception as e:
                st.error(f"Invalid seed: {e}")
        st.checkbox("Randomize who starts (New game)", key="random_start")

        st.divider()
        st.subheader("Training (Self-Play)")
        st.slider("Episodes per run", 10, 20000, key="episodes", value=500, step=10)
        st.slider("Min exploration during training ε_min", 0.0, 1.0, key="epsilon_min", value=0.05, step=0.01)
        st.slider("Min learning rate α_min", 0.0, 1.0, key="alpha_min", value=0.05, step=0.01)
        st.slider("Learning rate α", 0.0, 1.0, key="alpha", value=0.5, step=0.05)
        st.slider("Discount γ", 0.0, 1.0, key="gamma", value=0.99, step=0.01)
        st.slider("Exploration during training ε", 0.0, 1.0, key="epsilon_train", value=1.0, step=0.05)
        colA, colB = st.columns(2)
        with colA:
            if st.button("Train now"):
                cfg = QLearningConfig(
                    alpha=float(st.session_state.alpha),
                    gamma=float(st.session_state.gamma),
                    epsilon=float(st.session_state.epsilon_train),
                )
                total = int(st.session_state.episodes)
                progress = st.progress(0)
                status = st.empty()

                def progress_cb(done, total):
                    pct = int(done / total * 100)
                    progress.progress(pct)
                    status.text(f"Training episodes: {done}/{total} ({pct}%)")

                # Use the configured RNG seed for reproducible training
                seed = int(st.session_state.get("rng_seed", 42))
                eps_min = float(st.session_state.get("epsilon_min", 0.05))
                a_min = float(st.session_state.get("alpha_min", 0.05))
                # symmetry augmentation enabled
                self_play_train(st.session_state.Q, total, cfg, seed=seed, progress_callback=progress_cb, epsilon_min=eps_min, alpha_min=a_min, symmetry=True)
                progress.progress(100)
                status.text(f"Training completed: {total}/{total} (100%)")
                st.success(f"Trained for {st.session_state.episodes} episodes.")
        with colB:
            if st.button("Reset Q-table"):
                reset_q()
                st.warning("Q-table cleared (AI back to tabula rasa).")

        st.divider()
        st.subheader("Save / Load")
        st.download_button("Download Q-table (JSON)", data=save_q_json(), file_name="q_table.json")
        uploaded = st.file_uploader("Load Q-table (JSON)", type=["json"])
        if uploaded:
            load_q_json(uploaded.getvalue().decode("utf-8"))

    st.divider()
    st.subheader("Auto-play")
    st.checkbox("Auto-play AI when it's their turn", key="auto_play")
    st.number_input("Auto-play delay (seconds)", min_value=0, max_value=30, value=int(st.session_state.get("auto_play_delay", 2)), key="auto_play_delay", step=1, format="%d")

    st.divider()
    st.subheader("Board")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("New game"):
            # Respect the random-start preference
            reset_board(randomize_first=bool(st.session_state.get("random_start", False)))
            # Reset auto-play timer for potential AI-first games
            st.session_state["ai_auto_start_time"] = time.time()
    with c2:
        st.checkbox("Explain AI’s next move (show Q)", key="show_explain", value=True)

def render_board():
    st.subheader("Tic-Tac-Toe")
    board = st.session_state.board
    cur = st.session_state.current_player

    # Outcome banner
    w = winner(board)
    if w:
        st.success(f"🏁 {w} wins!")
    elif terminal(board):
        st.info("🤝 Draw!")

    # 3x3 grid of buttons
    rows = [st.columns(3) for _ in range(3)]
    for r in range(3):
        for c in range(3):
            idx = 3*r + c
            label = board[idx]
            disabled = label != EMPTY or terminal(board)
            with rows[r][c]:
                # Use on_click so the handler runs before Streamlit's rerun
                # and the updated session_state is reflected immediately.
                st.button(label if label != EMPTY else " ",
                          key=f"cell_{idx}",
                          disabled=disabled,
                          use_container_width=True,
                          on_click=handle_cell_click,
                          args=(idx,))

    st.caption(f"Current player: **{cur}**")

def render_explanation():
    if not st.session_state.show_explain:
        return
    board = st.session_state.board
    if terminal(board):
        return

    mode = st.session_state.mode
    if mode == "Human vs AI":
        human = st.session_state.human_plays_as
        ai = next_player(human)
        if st.session_state.current_player != ai:
            return
        player = ai
    else:
        # In 2 Humans mode, show Q for whoever is to move (teaching aid)
        player = st.session_state.current_player

    q = get_q_for_state(st.session_state.Q, board, player)
    legal = available_actions(board)
    if not legal:
        return

    # Choose best (greedy) to highlight
    best_val = max(q.get(a, 0.0) for a in legal)
    best_actions = [a for a in legal if q.get(a, 0.0) == best_val]

    # Epsilon used when playing (exploration probability)
    eps = float(st.session_state.get("epsilon_play", 1.0))
    n_legal = len(legal)

    st.subheader("🔍 How the AI evaluates this position")
    st.write("These are the **expected returns (Q)** the AI has learned for each legal move from this exact state.")
    st.write(f"Policy: epsilon-greedy (ε = {eps:.2f}). The AI will explore uniformly with probability ε and exploit the best action(s) with probability 1-ε.")

    data = []
    # Control buttons for AI moves
    col1, col2 = st.columns(2)
    with col1:
        # Preview sampled move control (does not advance the main RNG)
        if st.button("Preview sampled move"):
            # Use a separate RNG seeded from last_rng_seed + preview_counter for reproducibility;
            # do not modify the main RNG in session_state.
            seed_base = st.session_state.get("last_rng_seed")
            ctr = int(st.session_state.get("preview_counter", 0))
            try:
                if seed_base is not None:
                    tmp_rng = np.random.default_rng(int(seed_base) + ctr + 1)
                else:
                    tmp_rng = np.random.default_rng()
            except Exception:
                tmp_rng = np.random.default_rng()
            # Determine the player to sample for
            sample_action = epsilon_greedy_action(st.session_state.Q, board, player, eps, tmp_rng)
            st.session_state["preview_action"] = int(sample_action) if sample_action != -1 else None
            st.session_state["preview_counter"] = ctr + 1
            if sample_action != -1:
                rr, rc = divmod(sample_action, 3)
                st.success(f"Preview sampled move: action={sample_action} → row {rr+1}, col {rc+1}")
            else:
                st.info("Preview: no legal moves")
    
    with col2:
        if st.session_state.mode == "Human vs AI":
            human_as = st.session_state.human_plays_as
            ai_as = next_player(human_as)
            if st.session_state.current_player == ai_as and not terminal(st.session_state.board):
                # Use on_click so the move occurs during the event and updates session_state
                st.button("AI: Make move", on_click=maybe_ai_move)

    for a in sorted(legal):
        r, c = divmod(a, 3)
        # Probability under epsilon-greedy: exploration gives ε / n_legal to each action;
        # exploitation splits (1-ε) uniformly across best actions.
        prob = eps / n_legal
        if a in best_actions:
            prob += (1.0 - eps) / max(1, len(best_actions))
        data.append({
            "Action": a,
            "Cell # (1-based)": a + 1,
            "Row": r + 1,
            "Col": c + 1,
            "Q(a)": round(q.get(a, 0.0), 4),
            "P(move)": round(prob, 3),
            "Greedy?": "✅" if a in best_actions else "",
            "Preview?": "🔎" if st.session_state.get("preview_action") == a else "",
            "Last AI picked?": "🟢" if st.session_state.get("last_ai_action") == a else ""
        })

    st.dataframe(data, hide_index=True)
    st.caption("Higher Q means the move is expected to lead to a win (+1) and avoid a loss (−1) under optimal play, discounted by γ. 'P(move)' shows the epsilon-greedy probability the AI will select that action when playing.")

# ========== App ==========

st.set_page_config(page_title="Tic-Tac-Toe RL Demo", page_icon="🎓", layout="centered")
st.title("🎓 Tic-Tac-Toe with Q-Learning (Reinforcement Learning Demo)")
st.markdown(
    """
This interactive demo teaches **tabular Q-learning** using Tic-Tac-Toe.

**Suggested flow for class:**
1. Start with **Human vs AI** while the AI is untrained (ε=1.0 → random moves).
2. Click **Train now** (e.g., 5,000–20,000 episodes) with α≈0.5, γ≈0.99, ε≈0.2–0.5.
3. Set **AI exploration while playing ε → 0.0–0.1** and play again.
4. Toggle **Explain AI’s next move** to display the **Q-values** for each legal move.
    """
)

def run_app():
    init_state()
    render_controls()
    render_board()

    # Show the explanation before letting the AI act so students can inspect Q-values.
    render_explanation()

    # Auto-play: if enabled, and it's the AI's turn, trigger the move after the configured delay.
    try:
        if st.session_state.get("auto_play", False) and st.session_state.mode == "Human vs AI":
            human_as = st.session_state.human_plays_as
            ai_as = next_player(human_as)
            if st.session_state.current_player == ai_as and not terminal(st.session_state.board):
                delay = float(st.session_state.get("auto_play_delay", 2))
                t0 = st.session_state.get("ai_auto_start_time")
                if t0 is None:
                    # initialize timer
                    st.session_state["ai_auto_start_time"] = time.time()
                else:
                    elapsed = time.time() - t0
                    if elapsed >= delay:
                        maybe_ai_move()
    except Exception:
        # Don't let auto-play errors crash the app
        pass



if __name__ == "__main__":
    run_app()
