import pygame
import sys
import numpy as np
import random
import time
import os
from collections import deque
import pickle

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

# ------------------------------------------
# Globals & Colors (from your original code)
# ------------------------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

BLUE   = (0, 0, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)

# ------------------------------------------
# Connect4 Environment (unchanged)
# ------------------------------------------
class Connect4Env:
    def __init__(self):
        self.rows = ROW_COUNT
        self.cols = COLUMN_COUNT
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        
    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        return self.board.copy()
    
    def get_available_actions(self, board):
        return [col for col in range(self.cols) if board[0, col] == 0]
    
    def step(self, action, player):
        # place piece in the board
        if action not in self.get_available_actions(self.board):
            return self.board.copy(), -10, True, {"error": "Invalid move"}
        for row in range(self.rows-1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break
        # check win
        if self.check_win(self.board, player):
            return self.board.copy(), 1, True, {"winner": player}
        # check draw
        if len(self.get_available_actions(self.board)) == 0:
            return self.board.copy(), 0, True, {"winner": 0}
        # else ongoing
        return self.board.copy(), -0.01, False, {}
    
    def check_win(self, board, player):
        # Horizontal check
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if (board[r,c] == player and
                    board[r,c+1] == player and
                    board[r,c+2] == player and
                    board[r,c+3] == player):
                    return True
        # Vertical check
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if (board[r,c] == player and
                    board[r+1,c] == player and
                    board[r+2,c] == player and
                    board[r+3,c] == player):
                    return True
        # Diagonal (positive slope) check
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if (board[r,c] == player and
                    board[r+1,c+1] == player and
                    board[r+2,c+2] == player and
                    board[r+3,c+3] == player):
                    return True
        # Diagonal (negative slope) check
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if (board[r,c] == player and
                    board[r-1,c+1] == player and
                    board[r-2,c+2] == player and
                    board[r-3,c+3] == player):
                    return True
        return False

# ------------------------------------------
# Pygame board drawing function (unchanged)
# ------------------------------------------
def draw_board_pygame(board, screen):
    # Draw background grid
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            pygame.draw.rect(
                screen, BLUE,
                (c * SQUARESIZE, (r+1) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            )
            pygame.draw.circle(
                screen, BLACK,
                (c*SQUARESIZE + SQUARESIZE//2, (r+1)*SQUARESIZE + SQUARESIZE//2),
                RADIUS
            )
    # Draw pieces
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            piece = board[r][c]
            if piece == 1:
                color = RED
            elif piece == -1:
                color = YELLOW
            else:
                color = None
            if color:
                pygame.draw.circle(
                    screen, color,
                    (c*SQUARESIZE + SQUARESIZE//2, (r+1)*SQUARESIZE + SQUARESIZE//2),
                    RADIUS
                )
    pygame.display.update()

# ------------------------------------------
# Improved Network Architecture with CNN & Residual Blocks
# ------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    """
    CNN based network with residual blocks.
    Accepts board state of shape [batch, ROW_COUNT, COLUMN_COUNT].
    Outputs policy logits over columns and a scalar value.
    """
    def __init__(self, rows=ROW_COUNT, cols=COLUMN_COUNT, channels=64, num_res_blocks=2):
        super(AlphaZeroNet, self).__init__()
        self.rows = rows
        self.cols = cols
        
        # Initial convolution layer
        self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, cols)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x: [batch, ROW_COUNT, COLUMN_COUNT]
        x = x.unsqueeze(1)  # add channel dimension -> [batch, 1, rows, cols]
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

# ------------------------------------------
# Monte Carlo Tree Search (MCTS) Implementation (updated)
# ------------------------------------------
class MCTSNode:
    def __init__(self, parent=None, prior_prob=0.0):
        self.parent = parent
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

class MCTS:
    def __init__(self, net: AlphaZeroNet, n_simulations=50, c_puct=1.0):
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def search(self, env: Connect4Env, current_player, temp=1.0):
        """Perform MCTS from the current environment state, returning a distribution over columns."""
        root = MCTSNode(parent=None, prior_prob=1.0)

        # Evaluate root
        policy_probs, _ = self._policy_value_fn(env.board, current_player)
        legal_moves = env.get_available_actions(env.board)
        for a in range(COLUMN_COUNT):
            if a in legal_moves:
                child = MCTSNode(parent=root, prior_prob=policy_probs[a])
                root.children[a] = child
            else:
                policy_probs[a] = 0

        # Run simulations
        for _ in range(self.n_simulations):
            sim_env = self._copy_env(env)
            self._simulate(root, sim_env, current_player)

        # Build distribution from children
        counts = np.array([root.children[a].visit_count if a in root.children else 0
                           for a in range(COLUMN_COUNT)])
        if temp <= 1e-6:
            best_a = np.argmax(counts)
            probs = np.zeros_like(counts, dtype=np.float32)
            probs[best_a] = 1.0
        else:
            counts_exp = counts ** (1.0/temp)
            probs = counts_exp / np.sum(counts_exp)
        return probs

    def _simulate(self, node: MCTSNode, env: Connect4Env, current_player):
        # 1) Selection
        while node.children:
            action, node = self._select_child(node)
            env.step(action, current_player)
            if env.check_win(env.board, current_player):
                break
            if len(env.get_available_actions(env.board)) == 0:
                break
            current_player *= -1

        # 2) Expansion + Evaluation
        if (not env.check_win(env.board, current_player)
           and len(env.get_available_actions(env.board)) > 0):
            policy_probs, value_est = self._policy_value_fn(env.board, current_player)
            legal_moves = env.get_available_actions(env.board)
            for a in range(COLUMN_COUNT):
                if a in legal_moves:
                    child = MCTSNode(parent=node, prior_prob=policy_probs[a])
                    node.children[a] = child
                else:
                    policy_probs[a] = 0
        else:
            # Terminal state: win or draw
            if env.check_win(env.board, current_player):
                value_est = 1.0
            else:
                value_est = 0.0

        # 3) Backpropagation
        self._backprop(node, -value_est)

    def _select_child(self, node: MCTSNode):
        best_score = -float('inf')
        best_action, best_child = None, None
        for action, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior_prob * np.sqrt(node.visit_count + 1) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _backprop(self, node: MCTSNode, value):
        current = node
        sign = 1
        while current is not None:
            current.visit_count += 1
            current.total_value += value * sign
            current = current.parent
            sign = -sign

    def _policy_value_fn(self, board, player):
        # Player sees the board as themselves (multiplying by player)
        input_board = board.copy() * player
        # Use shape [1, ROW_COUNT, COLUMN_COUNT]
        inp = torch.FloatTensor(input_board).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(inp)
            policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return policy, value.item()

    def _copy_env(self, env: Connect4Env):
        import copy
        return copy.deepcopy(env)

# ------------------------------------------
# Self-play and Training Functions (updated)
# ------------------------------------------
def self_play(env: Connect4Env, net: AlphaZeroNet, n_games=5, n_sims=50, temp=1.0):
    """Generate training data from self-play."""
    mcts = MCTS(net, n_simulations=n_sims, c_puct=1.0)
    dataset = []
    for _ in range(n_games):
        env.reset()
        game_history = []
        current_player = 1
        done = False
        while not done:
            state = env.board.copy() * current_player  # shape: [rows, cols]
            pi = mcts.search(env, current_player, temp=temp)
            action = np.random.choice(range(COLUMN_COUNT), p=pi)
            game_history.append((state, pi, current_player))
            new_board, reward, done, info = env.step(action, current_player)
            if done:
                if "winner" in info and info["winner"] != 0:
                    winner = info["winner"]
                    for s, p, pl in game_history:
                        outcome = 1.0 if (pl == winner) else -1.0
                        dataset.append((s, p, outcome))
                else:
                    for s, p, pl in game_history:
                        dataset.append((s, p, 0.0))
            current_player *= -1
    return dataset

def train_alphazero(env: Connect4Env, net: AlphaZeroNet,
                    num_iterations=5, games_per_iter=10,
                    batch_size=32, lr=1e-3, n_sims=50):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for iteration in tqdm(range(num_iterations), desc="Training Iterations"):
        # Gather self-play data
        dataset = self_play(env, net, n_games=games_per_iter, n_sims=n_sims, temp=1.0)
        random.shuffle(dataset)
        states, pis, outcomes = zip(*dataset)
        
        # Here states are 2D boards: shape [rows, cols]
        states_tensor = torch.FloatTensor(states)  # [batch, ROW_COUNT, COLUMN_COUNT]
        pis_tensor    = torch.FloatTensor(pis)
        outcomes_tensor = torch.FloatTensor(outcomes).unsqueeze(1)
        
        # Process batches with a progress bar
        for start_idx in tqdm(range(0, len(states_tensor), batch_size), desc="Training Batches", leave=False):
            end_idx = start_idx + batch_size
            batch_states = states_tensor[start_idx:end_idx]
            batch_pis = pis_tensor[start_idx:end_idx]
            batch_outcomes = outcomes_tensor[start_idx:end_idx]
            
            logits, value = net(batch_states)
            policy = torch.softmax(logits, dim=1)
            
            policy_loss = -torch.mean(torch.sum(batch_pis * torch.log(policy + 1e-8), dim=1))
            value_loss  = torch.mean((value - batch_outcomes)**2)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tqdm.write(f"Iteration {iteration+1}/{num_iterations} done. Data size = {len(dataset)}")
    return net

# ------------------------------------------
# Helper: Save & Load Model
# ------------------------------------------
def save_alphazero_model(net: AlphaZeroNet, filename="alphazero_connect4.pth"):
    torch.save(net.state_dict(), filename)
    print(f"AlphaZero model saved to {filename}")

def load_alphazero_model(net: AlphaZeroNet, filename="alphazero_connect4.pth"):
    if os.path.exists(filename):
        net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        print(f"Loaded AlphaZero model from {filename}")
    else:
        print(f"No model found at {filename}; starting fresh.")

# ------------------------------------------
# Play Human vs. AlphaZero (updated for new network)
# ------------------------------------------
def play_human_vs_alphazero(env, net, screen, font, n_sims=300):
    mcts = MCTS(net, n_simulations=n_sims, c_puct=1.0)
    env.reset()
    current_player = 1
    game_over = False
    while not game_over:
        draw_board_pygame(env.board, screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Hovering logic for human
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                x_pos = event.pos[0]
                if current_player == 1:
                    pygame.draw.circle(screen, RED, (x_pos, SQUARESIZE//2), RADIUS)
                pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if current_player == 1:
                    x_pos = event.pos[0]
                    col = x_pos // SQUARESIZE
                    available_cols = env.get_available_actions(env.board)
                    if col in available_cols:
                        new_board, reward, done, info = env.step(col, current_player)
                        if done:
                            if "winner" in info and info["winner"] == 1:
                                label = font.render("You win!", 1, RED)
                            elif "winner" in info and info["winner"] == -1:
                                label = font.render("AI wins!", 1, YELLOW)
                            else:
                                label = font.render("Draw!", 1, WHITE)
                            screen.blit(label, (40, 10))
                            pygame.display.update()
                            time.sleep(3)
                            game_over = True
                        current_player *= -1
        # AI's turn
        if current_player == -1 and not game_over:
            pygame.time.wait(500)
            pi = mcts.search(env, current_player, temp=0.0)
            action = np.argmax(pi)
            new_board, reward, done, info = env.step(action, current_player)
            if done:
                if "winner" in info and info["winner"] == -1:
                    label = font.render("AI wins!", 1, YELLOW)
                elif "winner" in info and info["winner"] == 1:
                    label = font.render("You win!", 1, RED)
                else:
                    label = font.render("Draw!", 1, WHITE)
                screen.blit(label, (40, 10))
                pygame.display.update()
                time.sleep(3)
                game_over = True
            current_player *= -1
    env.reset()

# ------------------------------------------
# Main function with mode selection (focus on AlphaZero modes)
# ------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect 4 – Combined Agents")
    font = pygame.font.SysFont("monospace", 75)
    
    print("Choose game mode:")
    print("12: AlphaZero Connect4 (Play against trained model)")
    print("13: AlphaZero Connect4 (Extensive training mode)")
    
    mode = input("Enter mode (12 or 13): ").strip()
    while mode not in ["12", "13"]:
        mode = input("Enter mode (12 or 13): ").strip()
    mode = int(mode)
    
    if mode == 12:
        env = Connect4Env()
        net = AlphaZeroNet(rows=ROW_COUNT, cols=COLUMN_COUNT, channels=64, num_res_blocks=2)
        load_alphazero_model(net, "alphazero_connect4.pth")
        print("Loaded model. Starting human vs. AlphaZero play.")
        play_human_vs_alphazero(env, net, screen, font, n_sims=50)
        return
    
    if mode == 13:
        env = Connect4Env()
        net = AlphaZeroNet(rows=ROW_COUNT, cols=COLUMN_COUNT, channels=64, num_res_blocks=2)
        load_alphazero_model(net, "alphazero_connect4.pth")
        print("Starting extensive training. This may take a while...")
        # Adjust training hyperparameters as needed
        net = train_alphazero(
            env, net,
            num_iterations=50,     # More iterations for deeper training
            games_per_iter=500,    # More self-play games per iteration
            n_sims=200,            # Higher number of MCTS simulations
            batch_size=64,
            lr=1e-3
        )
        save_alphazero_model(net, "alphazero_connect4.pth")
        print("Extensive training complete and model saved.")
        return
    
    print("Mode not implemented. Exiting.")
    sys.exit()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")
    main()
