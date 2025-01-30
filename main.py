"""
Author: Janan Jahed
Filename: main.py
Description: The following code is the implmenetation of the bluff game
(doubt it variant). In this variant players take turns playing cards
face-down while declaring their rank, with opponents deciding whether
to accept or challenge the claim. The objective is to outsmart your
opponents by bluffing convincingly or catching them in a lie, with
the first player to run out of cards declared the winner.
The game is a zero-order and first-order theory of mind implementation played by
and AI vs. a human (you).

"""

import random
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CARD_RANKS = ['Ace', 'Jack', 'Queen', 'King']
INITIAL_CARDS_PER_PLAYER = 5

class Player:
    def __init__(self, name, theory_of_mind_order=0, is_human=False):
        self.name = name
        self.cards = []
        self.memory = []
        self.order = theory_of_mind_order
        self.is_human = is_human
        self.opponent_beliefs = {}
        logger.debug(f'Created player: {name} (Order: {theory_of_mind_order}, Human: {is_human})')

        if theory_of_mind_order == 1:
            for rank in CARD_RANKS:
                self.opponent_beliefs[rank] = {
                    'has_card': 0.5,
                    'will_bluff': 0.3
                }

    def receive_cards(self, cards):
        self.cards.extend(cards)
        logger.info(f'{self.name} received cards: {cards}')
        logger.info(f'{self.name} updated hand: {self.cards}')
        if self.is_human:
            print(f"\nyou received cards: {', '.join(cards)}")
            print(f"current hand: {self.cards}")

    def play_card(self, declared_rank):
        if not self.cards:
            logger.warning(f'{self.name} tried to play with empty hand')
            return 'pass', declared_rank

        if self.is_human:
            return self.human_play(declared_rank)
        
        return self.ai_play(declared_rank)

    def human_play(self, declared_rank):
        print(f"\nyour turn, {self.name}!")
        print(f"current declared rank: {declared_rank}")
        print(f"your cards: {', '.join(self.cards)}")
 
        while True:
            action = input("play a card (enter card) or pass: ").strip().capitalize()
            if action == 'Pass':
                logger.info(f'{self.name} passed')
                return 'pass', declared_rank
            if action in self.cards:
                self.cards.remove(action)
                logger.info(f'{self.name} played {action} as {declared_rank}')
                logger.info(f'{self.name} remaining cards: {self.cards}')
                return action, declared_rank
            print("Invalid card. Please choose from your hand or type 'pass'")

    def ai_play(self, declared_rank):
        try:
            if self.order == 0:
                return self.zero_order_play(declared_rank)
            else:
                return self.first_order_play(declared_rank)

        except IndexError:
            logger.error(f'{self.name} tried to play with empty hand')
            return 'pass', declared_rank

    def zero_order_play(self, declared_rank):
        """Zero-order agent uses a deterministic, rule-based approach"""
        if declared_rank in self.cards:
            played_card = declared_rank
        else:
            played_card = max(set(self.cards), key=self.cards.count)
        
        self.cards.remove(played_card)
        logger.info(f'{self.name} played {played_card} as {declared_rank}')
        logger.info(f'{self.name} remaining cards: {self.cards}')
        return played_card, declared_rank

    def first_order_play(self, declared_rank):
        """First-order agent models opponent's beliefs with adaptive learning"""
        self.update_beliefs(declared_rank)
        
        expected_opponent_challenge = self.opponent_beliefs[declared_rank]['will_bluff']
        
        if declared_rank in self.cards:
            if expected_opponent_challenge < 0.5:
                played_card = declared_rank
            else:
                played_card = declared_rank if random.random() < 0.7 else self.smart_bluff(declared_rank)
        else:
            played_card = self.smart_bluff(declared_rank)
        
        self.cards.remove(played_card)
        logger.info(f'{self.name} played {played_card} as {declared_rank}')
        logger.info(f'{self.name} remaining cards: {self.cards}')
        return played_card, declared_rank

    def smart_bluff(self, declared_rank):
        """Choose a bluff card based on opponent's likely cards"""
        safe_bluffs = [c for c in self.cards 
                      if self.opponent_beliefs.get(c, {}).get('has_card', 0.5) < 0.4]
        return random.choice(safe_bluffs) if safe_bluffs else random.choice(self.cards)

    def update_beliefs(self, declared_rank):
        """Update the belief matrix dynamically based on game history"""
        if self.order == 1:
            for rank in CARD_RANKS:
                if rank == declared_rank:
                    self.opponent_beliefs[rank]['has_card'] *= 0.7
                else:
                    self.opponent_beliefs[rank]['has_card'] = min(
                        self.opponent_beliefs[rank]['has_card'] + 0.1, 0.9
                    )

    def zero_order_challenge(self, declared_rank):
        """Zero-order agent challenges based on deterministic probability"""
        if not self.memory:
            return random.random() < 0.1
        bluff_rate = sum(1 for _, r in self.memory if r != declared_rank) / len(self.memory)
        return bluff_rate > 0.4


    def first_order_challenge(self, declared_rank):
        """First-order challenges based on belief matrix and observed play"""
        p_bluff = (1 - self.opponent_beliefs[declared_rank]['has_card']) * 0.8
        return p_bluff > 0.5

    def decide_challenge(self, declared_rank):
        if self.is_human:
            print(f"\n{self.name}, do you want to challenge the {declared_rank} claim?")
            decision = input("Challenge? (yes/no): ").strip().lower()
            logger.info(f'{self.name} decided to {"challenge" if decision == "yes" else "not challenge"}')
            return decision == 'yes'
            
        if self.order == 0:
            challenge = self.zero_order_challenge(declared_rank)
            logger.info(f'{self.name} (Zero-Order) challenge decision: {challenge}')
            return challenge
        else:
            challenge = self.first_order_challenge(declared_rank)
            logger.info(f'{self.name} (First-Order) challenge decision: {challenge}')
            return challenge

class BluffGame:
    def __init__(self, players):
        self.players = players
        self.deck = CARD_RANKS * 4
        self.current_rank = None
        self.last_passer = None
        self.winner = None
        self.history = []
        self.card_stack = []
        self.rounds = 0
        #Use of AI for formatting inspiration
        logger.info('\n\n=== Starting new game ===')
        
    def setup_game(self):
        random.shuffle(self.deck)
        logger.info('Initial hands:')
        for p in self.players:
            cards_to_deal = self.deck[:INITIAL_CARDS_PER_PLAYER]
            p.receive_cards(cards_to_deal)
            self.deck = self.deck[INITIAL_CARDS_PER_PLAYER:]
        logger.info('Cards dealt to players')

    def log_player_hands(self):
        logger.info('Current hands:')
        for player in self.players:
            logger.info(f'{player.name}: {player.cards}')

    def check_winner(self):
        for p in self.players:
            if not p.cards:
                self.winner = p
                logger.info(f'Game winner: {p.name}')
                return True
        return False
            
    def play_round(self):
        #Use of AI for formatting inspiration
        logger.info('\n--- Starting new round ---')
        self.log_player_hands()
        if self.check_winner():
            return
        
        self.rounds += 1
            
        starter = self.last_passer or self.players[0]
        self.current_rank = random.choice(CARD_RANKS)
        logger.info(f'New round declared rank: {self.current_rank}')
        current_player = starter
        pass_count = 0
        
        while True:
            if self.check_winner():
                return
                
            if not current_player.cards:
                current_player = self.next_player(current_player)
                continue
                
            logger.info(f"{current_player.name}'s turn")
            action = current_player.play_card(self.current_rank)
            
            if action[0] == 'pass':
                logger.info(f'{current_player.name} passed')
                self.last_passer = current_player
                pass_count += 1
                if pass_count >= len(self.players):
                    logger.info('All players passed')
                    return
                current_player = self.next_player(current_player)
                continue
                
            card, declared_rank = action
            self.history.append((current_player, card, declared_rank))
            self.card_stack.append(card)
            
            for challenger in self.players:
                if challenger != current_player and challenger.cards:
                    if challenger.decide_challenge(declared_rank):
                        logger.info(f'{challenger.name} challenges {current_player.name}')
                        return self.resolve_challenge(challenger, current_player, card, declared_rank)
                    
            current_player = self.next_player(current_player)
            pass_count = 0

    def next_player(self, current_player):
        idx = self.players.index(current_player)
        return self.players[(idx + 1) % len(self.players)]

    def resolve_challenge(self, challenger, player, card, declared_rank):
        logger.info(f'Resolving challenge: {card} vs {declared_rank}')
        logger.info(f'Cards in play: {self.card_stack}')
        
        if card != declared_rank:
            logger.info(f'Challenge succeeded! {challenger.name} wins')
            player.receive_cards(self.card_stack)
        else:
            logger.info(f'Challenge failed! {player.name} wins')
            challenger.receive_cards(self.card_stack)
            
        self.card_stack.clear()
        self.log_player_hands()
        return True

    def run_full_game(self):
        self.setup_game()
        while not self.winner:
            self.play_round()
            self.check_winner()
        return self.winner, self.rounds

class ExperimentRunner:
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(int))
        self.human_results = defaultdict(int)
        self.rounds_data = {
            'human_vs_zero': [],
            'human_vs_first': []
        }

    def run_agent_experiment(self, num_games=100):
        logger.info('Starting agent experiments')
        for _ in range(num_games):
            players = [
                Player("Zero-Order", 0),
                Player("First-Order", 1)
            ]
            game = BluffGame(players)
            winner, _ = game.run_full_game()
            self.results['agent_vs_agent'][winner.order] += 1

        # Save the results plot
        plt.figure(figsize=(6, 4))
        labels = ['Zero-Order', 'First-Order']
        values = [self.results['agent_vs_agent'][0], self.results['agent_vs_agent'][1]]
        plt.bar(labels, values, color=['blue', 'orange'])
        plt.title('Agent vs Agent Win Rates')
        plt.ylabel('Wins')
        plt.savefig('Finalrun.png')
        print("saved image")

    def run_human_experiment(self, games_per=5, num_participants=5):
        logger.info('Starting human experiments')
        for participant in range(1, num_participants + 1):
            for agent_type in ['zero', 'first']:
                for game_num in range(1, games_per + 1):
                    human = Player(f"Participant-{participant}", is_human=True)
                    agent = Player(f"{agent_type}-agent", 0 if agent_type == 'zero' else 1)

                    logger.info(f'Starting game {game_num} for participant {participant} vs {agent_type}-agent')
                    game = BluffGame([human, agent])
                    winner, rounds = game.run_full_game()
                    self.rounds_data[f'human_vs_{agent_type}'].append(rounds)

                    if winner.is_human:
                        self.human_results[agent_type] += 1

        with open('rounds_data.txt', 'w') as f:
            f.write("Human vs Zero-Order Rounds:\n")
            f.write(", ".join(map(str, self.rounds_data['human_vs_zero'])) + "\n")
            f.write("Human vs First-Order Rounds:\n")
            f.write(", ".join(map(str, self.rounds_data['human_vs_first'])) + "\n")
        print("saved txt file'")


if __name__ == "__main__":
    exp = ExperimentRunner()

    print("Running agent experiments...")
    exp.run_agent_experiment(num_games=100)

    print("\nStarting human experiments...")
    exp.run_human_experiment(games_per=5, num_participants=5)
