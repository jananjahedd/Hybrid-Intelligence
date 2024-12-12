"""
Author: Janan Jahed
Filename: main.py
Description: The following code is the implmenetation of the bluff game
(doubt it variant). In this variant players take turns playing cards
face-down while declaring their rank, with opponents deciding whether
to accept or challenge the claim. The objective is to outsmart your
opponents by bluffing convincingly or catching them in a lie, with
the first player to run out of cards declared the winner.
The game is a zero order theory of mind implementation played by
and AI vs. a human (you).

"""

import random
card_ranks = ['Ace', 'Jack', 'Queen', 'King']


class Player:
    """
    this class represents a player in the game.

    Attributes:
        name (str): The name of the player.
        is_human (bool): Whether the player is human or an AI agent.
        cards (list): The cards currently held by the player.
    """

    def __init__(self, name, is_human=False):
        """
        this function initializes a player with a name, type (human or AI),
        and an empty hand.

        Args:
            name (str): The name of the player.
            is_human (bool): True if the player is human; False for AI.
            Default is False.
        """
        self.name = name
        self.is_human = is_human
        self.cards = []

    def receive_cards(self, cards):
        """
        this function adds the given cards to the player's hand.

        Args:
            cards (list): A list of cards to add to the
            player's hand.
        """
        self.cards.extend(cards)
        if self.is_human:
            print(f"your current hand: {self.cards}")

    def play_card(self, declared_rank):
        """
        this function allows the player to play a card or pass their turn.

        Args:
            declared_rank (str): The rank of the card to be declared.

        Returns:
            tuple: A tuple containing the played card and the declared rank.
                   Returns ('pass', declared_rank) if the player passes.
        """
        declared_rank = declared_rank.capitalize()
        if declared_rank not in card_ranks:
            print(f"invalid rank '{declared_rank}'."
                  f"please choose from {card_ranks}.")
            return self.play_card(input("enter the rank again: "
                                        ).strip().capitalize())
        if self.is_human:
            print(f"\nyour hand: {self.cards}")
            card = input(f"select a card to play as the {declared_rank} "
                         f"card played? (or type 'pass'): ").strip()
            if card.lower() == 'pass':
                return 'pass', declared_rank
            elif card.capitalize() in self.cards:
                self.cards.remove(card.capitalize())
                return card.capitalize(), declared_rank
            else:
                print("invalid card. try again.")
                return self.play_card(declared_rank)
        else:
            # zero order agent - gusesses randomly
            card = random.choice(self.cards)
            self.cards.remove(card)
            return card, declared_rank

    def challenge(self, played_card, declared_rank):
        """
        this function decides whether to challenge the opponent's play.

        Args:
            played_card (str): The card played by the opponent.
            declared_rank (str): The rank declared by the opponent.

        Returns:
            bool: True if the player chooses to challenge; False otherwise.
        """
        if self.is_human:
            decision = input(f"do you want to challenge {declared_rank}? "
                             f"(yes/no): ").strip().lower()
            return decision == 'yes'
        else:
            # zero ordder agent - challenges randomly
            return random.choice([True, False])


class BluffGame:

    """
    this class manages the game logic for the "Doubt It" variant of Bluff.

    Attributes:
        players (list): A list of Player objects participating in the game.
        card_stack (list): The stack of cards played in the current round.
        current_rank (str): The rank of cards currently being played.
        last_player (Player): The last player to play a card.
        round_number (int): The current round number.
        pass_count (int): The number of consecutive passes in the current
        round.
        win_count (dict): A dictionary tracking the number of wins for each
        player.
    """

    def __init__(self, players):
        """
        this funcion initializes the game with the given players.

        Args:
            players (list): A list of Player objects.
        """
        self.players = players
        self.card_stack = []
        self.current_rank = None
        self.last_player = None
        self.round_number = 1
        self.pass_count = 0
        self.win_count = {player.name: 0 for player in players}

    def distribute_cards(self):
        """
        this function distributes cards evenly among the players at
        the start of the game.
        """
        all_cards = card_ranks * 4
        random.shuffle(all_cards)
        num_players = len(self.players)
        cards_per_player = len(all_cards) // num_players
        for player in self.players:
            player.receive_cards(all_cards[:cards_per_player])
            all_cards = all_cards[cards_per_player:]

    def start_round(self):
        """
        this function is for tarting a new round of the game.
        The players take turns to play cards/pass,
        and challenges are resolved when they occur.
        """
        self.card_stack = []
        self.current_rank = input("enter the starting rank for this round " +
                                  "(Ace/Jack/Queen/King): "
                                  ).strip().capitalize()
        if self.current_rank not in card_ranks:
            print(f"invalid rank '{self.current_rank}'. please choose:"
                  f"from {card_ranks}.")
            self.start_round()
            return
        self.pass_count = 0
        current_player_idx = 0

        while True:
            player = self.players[current_player_idx]
            print(f"\n{player.name}'s turn")
            card, declared_rank = player.play_card(self.current_rank)

            if card == 'pass':
                print(f"{player.name} passes.")
                self.pass_count += 1
                if self.pass_count >= len(self.players):
                    print("all players passed. starting a new round.")
                    self.current_rank = input("enter rank for the new round: "
                                              ).strip().capitalize()
                    if self.current_rank not in card_ranks:
                        print(f"invalid rank '{self.current_rank}'. "
                              f"please choose from {card_ranks}.")
                        self.start_round()
                        return
                    self.pass_count = 0
                current_player_idx = (current_player_idx + 1
                                      ) % len(self.players)
                continue

            self.card_stack.append((card, declared_rank))
            print(f"{player.name} plays {declared_rank}.")
            self.last_player = player

            next_player_idx = (current_player_idx + 1) % len(self.players)
            next_player = self.players[next_player_idx]

            if next_player.is_human:
                challenge = input(f"do you want to challenge the "
                                  f"{declared_rank} card played? (yes/no): "
                                  ).strip().lower() == 'yes'
            else:
                challenge = next_player.challenge(card, declared_rank)
                print(f"{next_player.name} decides to"
                      f" {'challenge' if challenge else 'not challenge'}.")

            if challenge:
                self.handle_challenge(next_player, card, declared_rank)
                return

            current_player_idx = next_player_idx

    def handle_challenge(self, challenger, played_card, declared_rank):
        """
        this function resolves a challenge by checking the validity of
        the last played card.

        Args:
            challenger (Player): The player who issued the challenge.
            played_card (str): The card being challenged.
            declared_rank (str): The rank declared with the card.
        """
        print(f"\n{challenger.name} challenges the play!")
        last_played_card, last_declared_rank = self.card_stack.pop()

        if last_played_card.lower() != declared_rank.lower():
            print(f"The challenge succeeds! {challenger.name} wins this round."
                  f"{self.last_player.name} played {last_played_card}.")
            self.card_stack = [last_played_card] + [
                card for card, _ in self.card_stack]
            self.last_player.receive_cards(self.card_stack)
            print(f"{self.last_player.name} gets the cards from this round")
            self.win_count[challenger.name] += 1
        else:
            print(f"The challenge fails! {challenger.name} loses this round. "
                  f"{self.last_player.name} played {last_played_card}.")
            self.card_stack = [last_played_card] + [
                card for card, _ in self.card_stack]
            challenger.receive_cards(self.card_stack)
            print(f"{challenger.name} receives all the cards from this round.")
            self.win_count[self.last_player.name] += 1

        self.card_stack.clear()

    def play_game(self):
        """
        this function manages the overall flow of the game, including rounds
        and end conditions.
        """
        print("Starting the game!")
        self.distribute_cards()
        while all(len(player.cards) > 0 for player in self.players):
            self.start_round()

        for player in self.players:
            if len(player.cards) == 0:
                print(f"\n{player.name} wins the game!")
                self.win_count[player.name] += 1
                break

        print("\nGame Over! Final Win Count:")
        for name, wins in self.win_count.items():
            print(f"{name}: {wins} wins")


def main():

    human_player = Player(name="Human", is_human=True)
    zero_order_agent = Player(name="Zero-Order AI")
    players = [human_player, zero_order_agent]

    game = BluffGame(players)
    game.play_game()


if __name__ == "__main__":
    main()
