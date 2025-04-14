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

LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON_START = 0.2
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
INITIAL_CONFIDENCE = 0.2

class TheoryOfMind:
    """Base class for Theory of Mind agents in the Bluff card game."""
    def __init__(self, name, is_human=False, learning_rate=LEARNING_RATE):
        self.name = name
        self.cards = []
        self.is_human = is_human
        self.learning_rate = learning_rate
        self.epsilon = EPSILON_START
        self.play_memory = []
        self.challenge_memory = []
    
    def receive_cards(self, cards):
        """Add cards to the player's hand."""
        self.cards.extend(cards)
        logger.info(f'{self.name} received cards: {cards}')
        logger.info(f'{self.name} updated hand: {self.cards}')
        if self.is_human:
            print(f"\nYou received cards: {', '.join(cards)}")
            print(f"Current hand: {self.cards}")
    
    def get_state(self, declared_rank):
        """
        Get a state representation.
        """
        in_hand = 1 if declared_rank in self.cards else 0
        declared_rank_idx = CARD_RANKS.index(declared_rank)
        return (len(self.cards), in_hand, declared_rank_idx)
    
    def update_beliefs(self, state, action):
        raise NotImplementedError
    
    def select_action(self, state):
        raise NotImplementedError
    
    def human_play(self, declared_rank):
        print(f"\nYour turn, {self.name}!")
        print(f"Declared rank: {declared_rank}")
        print(f"Your cards: {', '.join(self.cards)}")
        while True:
            action = input("Play a card (enter card) or type 'pass': ").strip().capitalize()
            if action == 'Pass':
                logger.info(f'{self.name} passed')
                self.play_memory.append((self.get_state(declared_rank), 'pass'))
                return 'pass', declared_rank
            if action in self.cards:
                self.cards.remove(action)
                logger.info(f'{self.name} played {action} as {declared_rank}')
                self.play_memory.append((self.get_state(declared_rank), action))
                return action, declared_rank
            print("Invalid card. Choose a valid card from your hand or type 'pass'.")
    
    def human_challenge(self, declared_rank):
        print(f"\n{self.name}, do you want to challenge the {declared_rank} claim?")
        decision = input("Challenge? (yes/no): ").strip().lower()
        challenge = (decision == 'yes')
        logger.info(f'{self.name} challenge decision: {"challenge" if challenge else "no challenge"}')
        state = self.get_state(declared_rank)
        self.challenge_memory.append((state, 'challenge' if challenge else 'no_challenge'))
        return challenge
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


class ToM0Player(TheoryOfMind):
    """
    Zero-order Theory of Mind player.
    
    Uses Q-learning and maintains a zero-order belief bp0(D) = {"C": p, "NC": 1-p}
    where p is the estimated probability that an opponent will challenge when a declared
    rank D is in force. For a truthful play the payoff is 1; for a bluff it is 2 - 3*p.
    """
    def __init__(self, name, is_human=False, learning_rate=LEARNING_RATE):
        super().__init__(name, is_human, learning_rate)
        self.play_q_table = defaultdict(lambda: 0.0)
        self.challenge_q_table = defaultdict(lambda: 0.0)
        self.bp0 = {rank: {"C": 0.5, "NC": 0.5} for rank in CARD_RANKS}
    
    def update_q_values(self, reward):
        for state, action in self.play_memory:
            old_q = self.play_q_table[(state, action)]
            new_q = old_q + self.learning_rate * (reward - old_q)
            self.play_q_table[(state, action)] = new_q
        self.play_memory = []
        self.decay_epsilon()
        logger.debug(f'{self.name} updated play Q-table with reward {reward}')
    
    def update_challenge_q_values(self, reward):
        for state, action in self.challenge_memory:
            old_q = self.challenge_q_table[(state, action)]
            new_q = old_q + self.learning_rate * (reward - old_q)
            self.challenge_q_table[(state, action)] = new_q
        self.challenge_memory = []
        self.decay_epsilon()
        logger.debug(f'{self.name} updated challenge Q-table with reward {reward}')
    
    def select_action(self, declared_rank):
        if self.is_human:
            return self.human_play(declared_rank)
        state = self.get_state(declared_rank)
        actions = self.cards + ['pass']
        if random.random() < self.epsilon:
            chosen = random.choice(actions)
            if chosen != 'pass':
                self.cards.remove(chosen)
            logger.debug(f'{self.name} (exploration) chooses {chosen} in state {state}')
            self.play_memory.append((state, chosen))
            return chosen, declared_rank
        q_values = {a: self.play_q_table[(state, a)] for a in actions}
        chosen = max(q_values, key=q_values.get)
        if chosen != 'pass':
            self.cards.remove(chosen)
        logger.debug(f'{self.name} (exploitation) chooses {chosen} in state {state} with Q-values {q_values}')
        self.play_memory.append((state, chosen))
        return chosen, declared_rank
    
    def decide_challenge(self, declared_rank):
        if self.is_human:
            return self.human_challenge(declared_rank)
        state = self.get_state(declared_rank)
        actions = ['challenge', 'no_challenge']
        if random.random() < self.epsilon:
            chosen = random.choice(actions)
            self.challenge_memory.append((state, chosen))
            return chosen == 'challenge'
        q_values = {a: self.challenge_q_table[(state, a)] for a in actions}
        chosen = max(q_values, key=q_values.get)
        self.challenge_memory.append((state, chosen))
        logger.info(f'{self.name} challenge decision Q-values: {q_values}, chosen: {chosen}')
        return chosen == 'challenge'
    
    def update_belief(self, declared_rank, challenged, my_action=None):
        state = declared_rank
        observed = {"C": 1.0, "NC": 0.0} if challenged else {"C": 0.0, "NC": 1.0}
        for outcome in ["C", "NC"]:
            old_val = self.bp0[state][outcome]
            self.bp0[state][outcome] = (1 - self.learning_rate) * old_val + self.learning_rate * observed[outcome]
        total = self.bp0[state]["C"] + self.bp0[state]["NC"]
        self.bp0[state]["C"] /= total
        self.bp0[state]["NC"] /= total
        logger.info(f'{self.name} updated bp0 for {declared_rank}: {self.bp0[state]}')


class ToM1Player(ToM0Player):
    """
    First-order Theory of Mind agent.
    
    In addition to bp0, maintains a first-order belief bp1 (updated based on its own action)
    and a confidence parameter c1. Also implements a Bayesian mental model of the opponent's type.
    
    The Bayesian model tracks an opponent’s hidden strategy (with four types) via:
      - A prior over opponent types ("strategy")
      - Conditional probability tables (CPTs) for actions given a type,
      - Nested beliefs about one's own strategy ("nested_beliefs").
    
    These elements are used to compute a Bayesian predicted challenge probability.
    """
    def __init__(self, name, is_human=False, learning_rate=LEARNING_RATE):
        super().__init__(name, is_human, learning_rate)
        self.bp1 = {rank: {"C": 0.5, "NC": 0.5} for rank in CARD_RANKS}
        self.c1 = INITIAL_CONFIDENCE
        self.tom_weight = 0.2
        self.opp_challenge_model = {rank: 0.5 for rank in CARD_RANKS}
        self.opp_bluffing_model = {rank: 0.5 for rank in CARD_RANKS}
        self.opp_behavior_stats = defaultdict(lambda: {'bluffed': 0, 'truthful': 0, 'total': 0})
        self.challenge_stats = {'success': 0, 'failure': 0, 'total': 0}
        self.card_count = {rank: 4 for rank in CARD_RANKS}
        self.opponent_action_history = []
        self.bayesian_model = {
            'strategy': {
                'aggressive_bluffer': 0.25,
                'conservative_player': 0.25,
                'opportunistic': 0.25,
                'random_player': 0.25
            },
            'action_given_strategy': {
                'bluff': {rank: {
                    'aggressive_bluffer': 0.7,
                    'conservative_player': 0.2,
                    'opportunistic': 0.5,
                    'random_player': 0.5
                } for rank in CARD_RANKS},
                'challenge': {rank: {
                    'aggressive_bluffer': 0.6,
                    'conservative_player': 0.3,
                    'opportunistic': 0.4,
                    'random_player': 0.5
                } for rank in CARD_RANKS}
            },
            'evidence_count': {
                'aggressive_bluffer': 1,
                'conservative_player': 1,
                'opportunistic': 1,
                'random_player': 1
            },
            'nested_beliefs': {
                'my_strategy': {
                    'aggressive_bluffer': 0.25,
                    'conservative_player': 0.25,
                    'opportunistic': 0.25,
                    'random_player': 0.25
                }
            }
        }
    
    def update_opp_type_beliefs(self, challenged):
        if challenged:
            likelihood = {
                'aggressive_bluffer': 0.8,
                'conservative_player': 0.3,
                'opportunistic': 0.5,
                'random_player': 0.5
            }
        else:
            likelihood = {
                'aggressive_bluffer': 0.2,
                'conservative_player': 0.7,
                'opportunistic': 0.5,
                'random_player': 0.5
            }
        for t in self.bayesian_model['strategy']:
            self.bayesian_model['strategy'][t] *= likelihood[t]
        total = sum(self.bayesian_model['strategy'].values())
        for t in self.bayesian_model['strategy']:
            self.bayesian_model['strategy'][t] /= total
        logger.info(f'{self.name} updated opponent type beliefs: {self.bayesian_model["strategy"]}')
    
    def update_bayesian_beliefs(self, action_type, declared_rank, observed_outcome):
        cpt = self.bayesian_model['action_given_strategy'][action_type][declared_rank]
        prior = self.bayesian_model['strategy']
        likelihoods = {}
        for strategy in prior:
            likelihood = cpt[strategy] if observed_outcome else (1 - cpt[strategy])
            likelihoods[strategy] = likelihood
        marginal = sum(prior[s] * likelihoods[s] for s in prior)
        if marginal > 0:
            for s in prior:
                posterior = (likelihoods[s] * prior[s]) / marginal
                self.bayesian_model['strategy'][s] = posterior
        lr = 0.05
        if action_type == 'bluff':
            for s in cpt:
                old_prob = cpt[s]
                observed = 1.0 if observed_outcome else 0.0
                cpt[s] = old_prob + lr * (observed - old_prob)
        self.update_nested_beliefs(action_type, declared_rank, observed_outcome)
        logger.debug(f"{self.name}'s updated Bayesian beliefs: {self.bayesian_model['strategy']}")
    
    def update_nested_beliefs(self, action_type, declared_rank, observed_outcome):
        if len(self.play_memory) > 0:
            last_state, last_action = self.play_memory[-1]
            my_last_bluff = (last_action != declared_rank and last_action != 'pass')
            nb = self.bayesian_model['nested_beliefs']['my_strategy']
            if my_last_bluff:
                nb['aggressive_bluffer'] += 0.02
                nb['conservative_player'] -= 0.02
            else:
                nb['conservative_player'] += 0.02
                nb['aggressive_bluffer'] -= 0.02
            total = sum(nb.values())
            for k in nb:
                nb[k] /= total
    
    def predict_opponent_action_bayesian(self, action_type, declared_rank, context=None):
        cpt = self.bayesian_model['action_given_strategy'][action_type][declared_rank]
        strategy_beliefs = self.bayesian_model['strategy']
        base_prob = sum(cpt[s] * strategy_beliefs[s] for s in strategy_beliefs)
        if context == 'have_card_in_hand' and action_type == 'bluff':
            base_prob *= 0.8
        elif context == 'have_card_in_hand' and action_type == 'challenge':
            base_prob *= 1.2
        nb = self.bayesian_model['nested_beliefs']['my_strategy']
        if nb['aggressive_bluffer'] > 0.6 and action_type == 'challenge':
            base_prob = min(0.95, base_prob * 1.3)
        return max(0.05, min(0.95, base_prob))
    
    def predict_opponent_challenge(self, declared_rank, is_truthful):
        challenge_prob = self.opp_challenge_model.get(declared_rank, 0.5)
        if is_truthful:
            challenge_prob *= 0.9
        else:
            challenge_prob = min(0.9, challenge_prob * 1.1)
        opponent_likely_has = self.card_count[declared_rank] > 0
        if opponent_likely_has and not is_truthful:
            challenge_prob *= 0.9
        context = 'have_card_in_hand' if declared_rank in self.cards else None
        bayesian_prob = self.predict_opponent_action_bayesian('challenge', declared_rank, context)
        combined_prob = (1 - self.tom_weight) * challenge_prob + self.tom_weight * bayesian_prob
        return combined_prob
    
    def calculate_action_value(self, action, declared_rank):
        state = self.get_state(declared_rank)
        q_value = self.play_q_table[(state, action)]
        if action == 'pass':
            return q_value
        is_truthful = (action == declared_rank)
        challenge_prob = self.predict_opponent_challenge(declared_rank, is_truthful)
        if is_truthful:
            expected_reward = 1
        else:
            expected_reward = challenge_prob * (-1) + (1 - challenge_prob) * 1
        combined_value = (1 - self.tom_weight) * q_value + self.tom_weight * expected_reward
        return combined_value
    
    def select_action(self, declared_rank):
        if self.is_human:
            return self.human_play(declared_rank)
        state = self.get_state(declared_rank)
        actions = self.cards + ['pass']
        if random.random() < self.epsilon:
            chosen = random.choice(actions)
            if chosen != 'pass':
                self.cards.remove(chosen)
            logger.debug(f'{self.name} (TOM1 exploration) chooses {chosen} in state {state}')
            self.play_memory.append((state, chosen))
            return chosen, declared_rank
        action_values = {a: self.calculate_action_value(a, declared_rank) for a in actions}
        chosen = max(action_values, key=action_values.get)
        if chosen != 'pass':
            self.cards.remove(chosen)
        logger.info(f'{self.name} (TOM1 reasoning) action values: {action_values} -> chosen: {chosen}')
        self.play_memory.append((state, chosen))
        return chosen, declared_rank
    
    def predict_bluff_probability(self, declared_rank):
        bluff_probability = self.opp_bluffing_model.get(declared_rank, 0.5)
        have_card = declared_rank in self.cards
        if have_card:
            bluff_probability = min(0.9, bluff_probability * 1.2)
        remaining = self.card_count[declared_rank]
        if remaining == 0 and not have_card:
            bluff_probability = 0.95
        elif remaining == 1 and not have_card:
            bluff_probability = 0.5
        state_key = (len(self.cards), declared_rank)
        if self.opp_behavior_stats[state_key]['total'] > 3:
            bluff_rate = self.opp_behavior_stats[state_key]['bluffed'] / self.opp_behavior_stats[state_key]['total']
            bluff_probability = 0.7 * bluff_rate + 0.3 * bluff_probability
        context = 'have_card_in_hand' if declared_rank in self.cards else None
        bayesian_prob = self.predict_opponent_action_bayesian('bluff', declared_rank, context)
        strategy_probs = self.bayesian_model['strategy']
        most_likely = max(strategy_probs, key=strategy_probs.get)
        if most_likely == 'aggressive_bluffer':
            combined_prob = 0.3 * bluff_probability + 0.7 * bayesian_prob
        elif most_likely == 'conservative_player':
            combined_prob = 0.5 * bluff_probability + 0.5 * bayesian_prob
        else:
            combined_prob = 0.5 * bluff_probability + 0.5 * bayesian_prob
        logger.debug(f'{self.name} bluff prediction: heuristic={bluff_probability:.2f}, bayesian={bayesian_prob:.2f}, combined={combined_prob:.2f}')
        return combined_prob
    
    def decide_challenge(self, declared_rank):
        if self.is_human:
            return self.human_challenge(declared_rank)
        state = self.get_state(declared_rank)
        if random.random() < self.epsilon:
            chosen = random.choice(['challenge', 'no_challenge'])
            self.challenge_memory.append((state, chosen))
            return chosen == 'challenge'
        bluff_probability = self.predict_bluff_probability(declared_rank)
        challenge_ev = bluff_probability * 1 + (1 - bluff_probability) * (-1)
        no_challenge_ev = 0
        q_challenge = self.challenge_q_table[(state, 'challenge')]
        q_no_challenge = self.challenge_q_table[(state, 'no_challenge')]
        strategy_probs = self.bayesian_model['strategy']
        dominant_strategy = max(strategy_probs, key=strategy_probs.get)
        strategy_confidence = max(strategy_probs.values())
        bayesian_adjustment = 0
        if dominant_strategy == 'aggressive_bluffer' and strategy_confidence > 0.4:
            bayesian_adjustment = 0.2
        elif dominant_strategy == 'conservative_player' and strategy_confidence > 0.4:
            bayesian_adjustment = -0.2
        else:
            bayesian_adjustment = 0
        combined_ev = (1 - self.tom_weight) * (q_challenge - q_no_challenge) \
                      + self.tom_weight * (challenge_ev - no_challenge_ev) \
                      + bayesian_adjustment
        decision = combined_ev > 0
        self.challenge_memory.append((state, 'challenge' if decision else 'no_challenge'))
        logger.info(f'{self.name} (TOM1 challenge) with bp0_C={self.bp0[declared_rank]["C"]:.2f}, bayesian p={self.predict_opponent_challenge(declared_rank, declared_rank in self.cards):.2f}, ev0={2*self.bp0[declared_rank]["C"]-1:.2f}, bayesian_ev={2*self.predict_opponent_challenge(declared_rank, declared_rank in self.cards)-1:.2f}, c1={self.c1:.2f}, adjustment={bayesian_adjustment:.2f} gives combined EV={combined_ev:.2f} and decides {"to challenge" if decision else "not to challenge"}.')
        return decision
    
    def update_belief(self, declared_rank, challenged, my_action=None):
        state = declared_rank
        observed_bp0 = {"C": 1.0, "NC": 0.0} if challenged else {"C": 0.0, "NC": 1.0}
        for outcome in ["C", "NC"]:
            old_val = self.bp0[state][outcome]
            self.bp0[state][outcome] = (1 - self.learning_rate) * old_val + self.learning_rate * observed_bp0[outcome]
        total = self.bp0[state]["C"] + self.bp0[state]["NC"]
        self.bp0[state]["C"] /= total
        self.bp0[state]["NC"] /= total
        if my_action is not None:
            target = "NC" if my_action == declared_rank else "C"
            observed_bp1 = {"C": 1.0 if target=="C" else 0.0, "NC": 1.0 if target=="NC" else 0.0}
            for outcome in ["C", "NC"]:
                old_val = self.bp1[state][outcome]
                self.bp1[state][outcome] = (1 - self.learning_rate) * old_val + self.learning_rate * observed_bp1[outcome]
            total1 = self.bp1[state]["C"] + self.bp1[state]["NC"]
            self.bp1[state]["C"] /= total1
            self.bp1[state]["NC"] /= total1
        p_bayes = self.predict_opponent_action_bayesian('challenge', declared_rank)
        predicted = "C" if p_bayes > 0.5 else "NC"
        actual = "C" if challenged else "NC"
        if predicted == actual:
            self.c1 = min(0.8, (1 - self.learning_rate) * self.c1 + self.learning_rate)
        else:
            self.c1 = max(0.2, (1 - self.learning_rate) * self.c1)
        self.update_opp_type_beliefs(challenged)
        logger.info(f'{self.name} updated for {declared_rank}: bp0={self.bp0[state]}, bp1={self.bp1[state]}, c1={self.c1:.2f}, Bayesian strategy={self.bayesian_model["strategy"]}')


class BluffGame:
    """Game environment for the Bluff card game."""
    def __init__(self, players):
        self.players = players
        self.deck = CARD_RANKS * 4
        self.current_rank = None
        self.last_passer = None
        self.winner = None
        self.history = []
        self.card_stack = []
        self.rounds = 0
        self.discarded_cards = []
        logger.info('\n\n=== Starting new game ===')
    
    def setup_game(self):
        random.shuffle(self.deck)
        logger.info('Dealing initial hands:')
        num_players = len(self.players)
        cards_per_player = len(self.deck) // num_players
        for p in self.players:
            cards_to_deal = self.deck[:cards_per_player]
            p.receive_cards(cards_to_deal)
            self.deck = self.deck[cards_per_player:]
        logger.info('Cards dealt to players')
    
    def log_player_hands(self):
        logger.info('Current hands:')
        for p in self.players:
            logger.info(f'{p.name}: {p.cards}')
    
    def check_winner(self):
        for p in self.players:
            if not p.cards:
                self.winner = p
                logger.info(f'Game winner: {p.name}')
                return True
        return False
    
    def play_round(self):
        logger.info('\n--- Starting new round ---')
        self.log_player_hands()
        if self.check_winner():
            return
        self.rounds += 1
        for p in self.players:
            p.play_memory = []
            p.challenge_memory = []
        self.current_rank = random.choice(CARD_RANKS)
        logger.info(f'New round declared rank: {self.current_rank}')
        current_player = self.last_passer or self.players[0]
        pass_count = 0
        while True:
            if self.check_winner():
                return
            if not current_player.cards:
                current_player = self.next_player(current_player)
                continue
            logger.info(f"{current_player.name}'s turn (declared rank: {self.current_rank})")
            action, declared = current_player.select_action(self.current_rank)
            if action == 'pass':
                logger.info(f'{current_player.name} passed.')
                self.last_passer = current_player
                pass_count += 1
                if pass_count >= len(self.players):
                    logger.info('All players passed. Ending round with no challenge.')
                    if self.card_stack:
                        self.discarded_cards.extend(self.card_stack)
                        self.card_stack.clear()
                    return
                current_player = self.next_player(current_player)
                continue
            self.history.append((current_player, action, declared))
            self.card_stack.append(action)
            is_bluff = (action != declared)
            challenge_occurred = False
            for opponent in self.players:
                if opponent != current_player and opponent.cards:
                    challenge = opponent.decide_challenge(declared)
                    if challenge:
                        logger.info(f'{opponent.name} challenges {current_player.name}!')
                        challenge_occurred = True
                        outcome = self.resolve_challenge(opponent, current_player, action, declared)
                        for p in self.players:
                            if isinstance(p, ToM1Player) and p != current_player:
                                p.update_belief(declared, challenged=True)
                        if isinstance(current_player, ToM1Player):
                            current_player.update_belief(declared, challenged=True, my_action=action)
                        return
            if not challenge_occurred:
                for p in self.players:
                    if isinstance(p, ToM1Player) and p == current_player:
                        p.update_belief(declared, challenged=False, my_action=action)
                    elif isinstance(p, ToM1Player):
                        p.update_belief(declared, challenged=False)
            current_player = self.next_player(current_player)
            pass_count = 0
    
    def next_player(self, current_player):
        idx = self.players.index(current_player)
        return self.players[(idx + 1) % len(self.players)]
    
    def resolve_challenge(self, challenger, player, card, declared):
        logger.info(f'Resolving challenge: played {card} vs declared {declared}')
        logger.info(f'Cards in play: {self.card_stack}')
        if card != declared:
            logger.info(f'Challenge succeeded! {challenger.name} wins the round.')
            player.receive_cards(self.card_stack)
            outcome = {'challenger': True, 'player': False}
        else:
            logger.info(f'Challenge failed! {player.name} wins the round.')
            challenger.receive_cards(self.card_stack)
            outcome = {'challenger': False, 'player': True}
        self.discarded_cards.extend(self.card_stack)
        self.card_stack.clear()
        self.log_player_hands()
        return outcome
    
    def update_q_values(self, challenger, player, outcome):
        for p in [challenger, player]:
            reward = 1 if (p == challenger and outcome['challenger']) or (p == player and outcome['player']) else -1
            p.update_q_values(reward)
            p.update_challenge_q_values(reward)
    
    def run_full_game(self):
        self.setup_game()
        while not self.winner:
            self.play_round()
            self.check_winner()
        return self.winner, self.rounds


class ExperimentRunner:
    """Run experiments with different player configurations."""
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(int))
        self.human_results = defaultdict(int)
        self.rounds_data = {'agent_vs_agent': []}
    
    def run_agent_experiment(self, num_games=10000):
        logger.info('Starting agent vs. agent experiments')
        winners = []
        rounds_list = []
        win_rates_over_time = {'zero': [], 'first': []}
        rounds_per_100 = []
        current_rounds = []
        zero_wins_interval = 0
        first_wins_interval = 0
        for game_num in range(1, num_games + 1):
            players = [ToM0Player("Zero-Order"), ToM1Player("First-Order")]
            game = BluffGame(players)
            winner, rounds = game.run_full_game()
            winner_type = 1 if isinstance(winner, ToM1Player) else 0
            winners.append(winner_type)
            self.results['agent_vs_agent'][winner_type] += 1
            rounds_list.append(rounds)
            current_rounds.append(rounds)
            if winner_type == 0:
                zero_wins_interval += 1
            else:
                first_wins_interval += 1
            if game_num % 100 == 0:
                zero_rate = zero_wins_interval / 100
                first_rate = first_wins_interval / 100
                win_rates_over_time['zero'].append(zero_rate)
                win_rates_over_time['first'].append(first_rate)
                rounds_per_100.append(sum(current_rounds) / len(current_rounds))
                zero_wins_interval = 0
                first_wins_interval = 0
                current_rounds = []
                logger.info(f"Games {game_num-99}-{game_num}: Zero-Order win rate: {zero_rate:.2f}, First-Order win rate: {first_rate:.2f}")
        self._plot_win_rates(self.results['agent_vs_agent'])
        self._plot_learning_curve(rounds_list, num_games)
        self._plot_win_rates_over_time(win_rates_over_time, num_games)
        self._plot_avg_rounds(rounds_per_100, num_games)
        self._plot_cumulative_win_rates(winners, num_games)
        self._plot_agent_learning_curve(win_rates_over_time)
    
    def run_human_experiment(self, games_per=5, num_participants=5):
        logger.info('Starting human vs. agent experiments')
        total_games = 0
        human_wins = 0
        agent_wins = 0
        rounds_list = []
        for participant in range(1, num_participants + 1):
            for agent_type in ['zero', 'first']:
                for game_num in range(1, games_per + 1):
                    human = ToM0Player(f"Participant-{participant}", is_human=True)
                    if agent_type == 'zero':
                        agent = ToM0Player("zero-agent")
                    else:
                        agent = ToM1Player("first-agent")
                    game = BluffGame([human, agent])
                    winner, rounds = game.run_full_game()
                    rounds_list.append(rounds)
                    total_games += 1
                    if winner.name.startswith("Participant"):
                        human_wins += 1
                    else:
                        agent_wins += 1
                    print(f"Game result: Winner = {winner.name}, Rounds = {rounds}")
        human_win_rate = human_wins / total_games if total_games > 0 else 0
        agent_win_rate = agent_wins / total_games if total_games > 0 else 0
        print(f"\nHuman win rate: {human_win_rate:.2f}, Agent win rate: {agent_win_rate:.2f}")
        
        with open("human_experiment_results.txt", "w") as f:
            f.write("Human vs Agent Experiment Results\n")
            f.write(f"Total games: {total_games}\n")
            f.write(f"Human wins: {human_wins}\n")
            f.write(f"Agent wins: {agent_wins}\n")
            f.write(f"Human win rate: {human_win_rate:.2f}\n")
            f.write(f"Agent win rate: {agent_win_rate:.2f}\n")
            f.write("Rounds per game:\n")
            for r in rounds_list:
                f.write(f"{r}\n")
        print("Results saved to human_experiment_results.txt")
    
    def _plot_win_rates(self, win_data):
        labels = ['ToM0', 'ToM1']
        win_counts = [win_data[0], win_data[1]]
        plt.figure()
        plt.bar(labels, win_counts)
        plt.title('Win Rates')
        plt.xlabel('Agent Type')
        plt.ylabel('Wins')
        plt.savefig('win_rates.png')
        plt.show()
    
    def _plot_learning_curve(self, rounds_data, num_games):
        plt.figure()
        plt.plot(rounds_data)
        plt.title('Game Length Learning Curve')
        plt.xlabel('Game Number')
        plt.ylabel('Rounds')
        plt.savefig('learning_curve.png')
        plt.show()
    
    def _plot_win_rates_over_time(self, win_rates, num_games):
        plt.figure()
        plt.plot(win_rates['zero'], label='ToM0')
        plt.plot(win_rates['first'], label='ToM1')
        plt.title('Win Rates Over Time')
        plt.xlabel('100 Game Intervals')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.savefig('win_rates_over_time.png')
        plt.show()
    
    def _plot_avg_rounds(self, rounds_per_100, num_games):
        plt.figure()
        plt.plot(rounds_per_100)
        plt.title('Average Rounds per 100 Games')
        plt.xlabel('Interval')
        plt.ylabel('Average Rounds')
        plt.savefig('avg_rounds.png')
        plt.show()
    
    def _plot_cumulative_win_rates(self, winners, num_games):
        cumulative = np.cumsum(winners)
        plt.figure()
        plt.plot(cumulative)
        plt.title('Cumulative Win Rates (ToM1 Wins)')
        plt.xlabel('Game Number')
        plt.ylabel('Cumulative Wins for ToM1')
        plt.savefig('cumulative_win_rates.png')
        plt.show()
    
    def _plot_agent_learning_curve(self, win_rates):
        intervals = range(1, len(win_rates['zero']) + 1)
        plt.figure()
        plt.plot(intervals, win_rates['zero'], color='orange', label='ToM0')
        plt.plot(intervals, win_rates['first'], color='blue', label='ToM1')
        plt.xlabel('Interval (per 100 games)')
        plt.ylabel('Win Rate')
        plt.title('Learning Curve of Agents')
        plt.legend()
        plt.savefig('agent_learning_curve.png')
        plt.show()

if __name__ == "__main__":
    runner = ExperimentRunner()
    print("Select mode:")
    print("1: Agent vs Agent experiment")
    print("2: Human vs Agent experiment")
    mode = input("Enter mode (1 or 2): ").strip()
    if mode == "1":
        num_games = int(input("Enter number of games for the agent experiment: "))
        runner.run_agent_experiment(num_games)
    elif mode == "2":
        games_per = int(input("Enter number of games per participant: "))
        num_participants = int(input("Enter number of participants: "))
        runner.run_human_experiment(games_per, num_participants)
    else:
        print("Invalid mode. Exiting.")
