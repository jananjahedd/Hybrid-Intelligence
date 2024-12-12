# Hybrid-Intelligence - Bluff game, variant Doubt it
(At the moment for the zero-order theory of mind implementation everything is in a main.py file but for the final submission I will seperate the classes and use inheritance for code readability)



## Overview
The Bluff Game, or "Doubt It," is a strategic card game in which the players aim to outplay their rivals by bluffing convincingly or catching them lying/bluffing. This code implementation features a human player competing against a Zero-Order AI agent. The end goal is to be the first player to run out of cards by successfully playing cards or detecting the opponent's bluffs.


## Game Description
### Concept
- **Players**: Human vs. AI (Zero-Order Agent).
- **Objective**: Run out of cards before your opponent by:
  - Bluffing convincingly (playing a card and declaring a potentially false rank).
  - Detecting when your opponent is bluffing and issuing a challenge.

### How It Works
- Players take turns playing their faced-down cards. At the beginning of the game, the first player declares a rank. The players should put a true card that is the same as the starting rank or bluff their way out of the round. 
- Opponents can choose to accept the play or challenge the players guess.
- If they challenged:
  - The actual card played is revealed to verify if it is the same as the declared rank.
  - Based on the result, the challenger or the player collects all the cards played in the current round.
- The game continues until one player runs out of cards. This player will be the winner of the game. 


## Features
- **Zero-Order Theory of Mind AI**:
  - The AI does not model or infer the mental state of the human player.
  - It selects moves randomly without any strategy based on the human player's behavior.

- **Human-Agent Interaction**:
  - Allows human players to challenge the AI’s moves.
  - Provides input prompts and displays the current state of the game.


## Rules
### Game Setup
1. A standard deck of cards (Ace, Jack, Queen, King; 4 copies of each).
2. Cards are shuffled and distributed between the 2 players equally.


## AI Behavior
- The AI (Zero-Order Agent) performs actions based solely on its own cards.
- Decision-making:
  - Randomly selects a card to play.
  - Randomly decides whether to challenge the human player's declared rank.


## How to Play
1. Run the game script (`main.py`).
2. The game will prompt the human player for their actions during their turn.
3. Follow the prompts to:
   - Declare a card rank.
   - Play a card or pass.
   - Decide whether to challenge the AI's play.
4. Observe the game's feedback to see the result of each action.
5. Continue playing rounds until one player wins.


## File Structure
- **main.py**: Entry point of the game. Contains the logic for setting up and running the game.
- **Classes**:
  - `Player`: Represents a human or AI player.
  - `BluffGame`: Manages game logic, rounds, and win conditions.


## Example Gameplay
1. **Start Round**:
   - Human declares the starting rank as "Ace."
   - Human plays a card and declares it as "Ace."
2. **AI’s Turn**:
   - AI randomly plays a card and declares it as "Ace."
   - Human decides to challenge.
3. **Challenge Resolution**:
   - If AI’s card is not "Ace," AI collects all cards.
   - If AI’s card is "Ace," Human collects all cards.
4. The game continues until one player runs out of cards.


## Requirements
- Python 3.6+
- No external libraries required.


## How to Run
1. Clone the repository.
2. Navigate to the folder containing `main.py`.
3. Run the script:
   ```bash
   python main.py
   ```


## Future Enhancements
- Add first-order Theory of Mind AI to predict and counter human strategies.
- Use OOP principles to fix readability


