# Tic-Tac-Toe Q-Learning Demo

This is a small interactive Streamlit app used for teaching tabular Q-learning with Tic-Tac-Toe.

Files
- `tictactoe_app.py` - main Streamlit app.

Quick start (PowerShell)

```powershell
streamlit run tic_tac_toe_rl_app.py
```

What to try in class
- Mode: pick "Human vs AI" or "2 Humans".
- Explain AI’s next move: enables the Q-table explanation panel so students can inspect Q-values before the AI acts.
- RNG seed: set an integer seed and click **Reseed RNG** to make stochastic behavior reproducible.
- Train now: run the `Train now` button to perform self-play training for the specified number of episodes. Consider setting episodes high (5k–20k) for a stronger policy; add a progress bar if you want to show progress.

Notes
- The Q-table is stored in-memory in the Streamlit session; use the Download/Upload controls to export/import a Q-table as JSON.
- The app intentionally shows the AI's Q-values before it acts; click "AI: Make move" when you are ready for the AI to play.

