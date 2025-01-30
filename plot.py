import matplotlib.pyplot as plt
from collections import defaultdict

log_file = "human.log"

human_wins = defaultdict(int)
agent_wins = defaultdict(int)
processed_games = set()

def parse_log_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    parsing = False
    current_agent = ""
    current_game = 0

    for line in lines:
        if "Starting game" in line and "participant" in line:
            parsing = True
            current_game += 1 
            if "zero-agent" in line:
                current_agent = "Zero-Order"
            elif "first-agent" in line:
                current_agent = "First-Order"

        if parsing and "Game winner:" in line:
            winner = line.split("Game winner:")[1].strip()
            game_key = (current_game, current_agent)
            # for avoiding duplicate counts in log file
            if game_key not in processed_games:  
                processed_games.add(game_key)
                if winner.startswith("Participant"):
                    human_wins[current_agent] += 1
                else:
                    agent_wins[current_agent] += 1

def plot_results():
    agents = ["Zero-Order", "First-Order"]

    human_results = [human_wins[agent] for agent in agents]
    agent_results = [agent_wins[agent] for agent in agents]

    bar_width = 0.35
    x = range(len(agents))

    plt.bar(x, human_results, width=bar_width, label="Humans", color="blue")
    plt.bar([p + bar_width for p in x], agent_results, width=bar_width, label="Agents", color="orange")

    plt.xlabel("Agent Type")
    plt.ylabel("Wins")
    plt.title("Human vs Agent Performance")
    plt.xticks([p + bar_width / 2 for p in x], agents)
    plt.legend()

    plt.tight_layout()
    plt.show()

parse_log_file(log_file)
plot_results()

print("Results:")
print(f"Zero-Order - Humans: {human_wins['Zero-Order']}, Agents: {agent_wins['Zero-Order']}")
print(f"First-Order - Humans: {human_wins['First-Order']}, Agents: {agent_wins['First-Order']}")
print(f"Total games for Zero-Order: {sum(human_wins['Zero-Order'] + agent_wins['Zero-Order'])}")
print(f"Total games for First-Order: {sum(human_wins['First-Order'] + agent_wins['First-Order'])}")
