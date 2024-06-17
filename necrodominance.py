from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from typing import FrozenSet
from copy import copy
import random


class Mana(Enum):
    WHITE = 0
    BLACK = 1
    RED = 2
    GREEN = 3
    BLUE = 5
    COLORLESS = 6
    ANY = 7


@dataclass(repr=False)
class Card:
    is_artifact: bool = False
    is_land: bool = False
    initial_mana: bool = False
    is_protection: bool = False
    fixes: int = 0
    costs_mana: FrozenSet[tuple[Mana, int]] = frozenset([])
    makes_mana: FrozenSet[tuple[Mana, int]] = frozenset([])

    def mana_costs(self) -> dict[Mana, int]:
        return {k: v for k, v in self.costs_mana}
    
    def mana_makes(self) -> dict[Mana, int]:
        return {k: v for k, v in self.makes_mana}
    
    def amount_of_mana_produced(self, m: Mana) -> int:
        mana_makes = self.mana_makes()
        return mana_makes.get(m, 0) + mana_makes.get(Mana.ANY, 0)

    def __repr__(self):
        return self.__class__.__name__
    

@dataclass
class SampleResult:
    protected: bool
    won: bool
    hand_history: list[list[Card]]

    def __repr__(self):
        return f"Protected: {self.protected}, Won: {self.won}\nHand History:\n{"\n".join([repr(hand) for hand in self.hand_history])}"


@dataclass(repr=False)
class Necrodominance(Card):
    costs_mana: FrozenSet[Mana] = frozenset([(Mana.BLACK, 3)])

    def __repr__(self):
        return "💀Necrodominance💀"


@dataclass(repr=False)
class Vault(Card):
    is_artifact: bool = True
    is_land: bool = True
    initial_mana: bool = True
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.BLACK, 1)])


@dataclass(repr=False)
class GemstoneMine(Card):
    is_land: bool = True
    initial_mana: bool = True
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 1)])


@dataclass(repr=False)
class Petal(Card):
    is_artifact: bool = True
    initial_mana: bool = True
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 1)])


@dataclass(repr=False)
class PactOfNegation(Card):
    is_protection: bool = True


@dataclass(repr=False)
class SummonersPact(Card):
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.RED, 1)])


@dataclass(repr=False)
class SerumPowder(Card):
    is_artifact: bool = True


@dataclass(repr=False)
class Beseech(Card):
    costs_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 1), (Mana.BLACK, 3)])

    def __repr__(self):
        return "👸Beseech👸"


@dataclass(repr=False)
class BorneOnAWind(Card): ...

@dataclass(repr=False)
class WeatherTheStorm(Card): ...



@dataclass(repr=False)
class CabalRitual(Card):
    initial_mana: bool = False
    costs_mana: FrozenSet[Mana] = frozenset([
        (Mana.ANY, 1), (Mana.BLACK, 1)
    ])
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.BLACK, 3)])



@dataclass(repr=False)
class DarkRitual(Card):
    initial_mana: bool = False
    costs_mana: FrozenSet[Mana] = frozenset([(Mana.BLACK, 1)])
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.BLACK, 3)])


@dataclass(repr=False)
class Manamorphose(Card):
    fixes: int = 2
    costs_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 2)])
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 2)])


@dataclass(repr=False)
class SimianSpiritGuide(Card):
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.RED, 1)])
    initial_mana: bool = True


@dataclass(repr=False)
class ElvishSpiritGuide(Card):
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.GREEN, 1)])
    initial_mana: bool = True


@dataclass(repr=False)
class ValakutAwakening(Card): ...


@dataclass(repr=False)
class Tendrils(Card): ...


@dataclass(repr=False)
class Chancellor(Card):
    is_protection: bool = True


@dataclass(repr=False)
class WildCantor(Card):
    initial_mana: bool = False
    costs_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 1)])
    makes_mana: FrozenSet[Mana] = frozenset([(Mana.ANY, 1)])


def mana_costs_covered(mana_pool: dict[Mana, int], mana_costs: dict[Mana, int]):
    consumed_mana = copy(mana_pool)
    mana_to_fulfill = copy(mana_costs)
    colored_mana_constraints = {
        k: v for k, v in mana_to_fulfill.items() if k != Mana.ANY
    }

    for mana, amount_required in colored_mana_constraints.items():
        generic_mana_available = consumed_mana.get(Mana.ANY, 0)
        colored_mana_available = consumed_mana.get(mana, 0)
        if amount_required > colored_mana_available + generic_mana_available:
            return False
        if amount_required > colored_mana_available:
            amount_required -= colored_mana_available
            consumed_mana[mana] = 0
            consumed_mana[Mana.ANY] -= amount_required
        else:
            consumed_mana[mana] -= amount_required
    
    generic_mana_to_fullfill = mana_to_fulfill.get(Mana.ANY, 0)
    return sum(consumed_mana.values()) >= generic_mana_to_fullfill

def mana_priority(x: Mana):
    return {
        Mana.COLORLESS: 0,
        Mana.WHITE: 1,
        Mana.BLUE: 2,
        Mana.RED: 3,
        Mana.GREEN: 4,
        Mana.BLACK: 5,
        Mana.ANY: 6
    }[x]


class ManaGenerator:
    def __init__(self, mana_priority: Mana = None):
        self.mana_priority = mana_priority
        self.land_drop_made = False

    def _can_cast(self, card: Card, mana_pool: dict[Mana, int]):
        if card.is_land and self.land_drop_made:
            return False
        return mana_costs_covered(mana_pool, card.mana_costs())

    def _cast(self, card: Card, mana_pool: dict[Mana, int]):
        if card.is_land:
            self.land_drop_made = True

        for mana, cost in { k: v for k, v in card.mana_costs().items() if k != Mana.ANY}.items():
            mana_used = 0
            generic_mana_needed = 0
            if mana in mana_pool:
                mana_available = mana_pool[mana]
                colored_mana_to_use = min(cost, mana_available)
                generic_mana_needed = cost - colored_mana_to_use
                mana_pool[mana] -= colored_mana_to_use
            if generic_mana_needed > 0:
                mana_pool[Mana.ANY] -= generic_mana_needed

        assert all([v >= 0 for v in mana_pool.values()])
        any_mana_left_to_spend = card.mana_costs().get(Mana.ANY, 0)
        if any_mana_left_to_spend > 0:
            for mana_type in sorted(mana_pool, key=mana_priority):
                if any_mana_left_to_spend <= 0:
                    break
                mana_left_of_type = mana_pool[mana_type]
                used_of_type = min(any_mana_left_to_spend, mana_left_of_type)
                mana_pool[mana_type] -= used_of_type
                any_mana_left_to_spend -= used_of_type
        
        assert all([v >= 0 for v in mana_pool.values()])

                    

        for mana, produced in card.makes_mana:
            if mana in mana_pool:
                mana_pool[mana] += produced
            else:
                mana_pool[mana] = produced

    def _run(self, initial_mana_cards: list[Card], initial_mana_pool: dict[Mana, int] = None):
        mana_pool = copy(initial_mana_pool) if initial_mana_pool is not None else {}
        current_cards = initial_mana_cards[:]
        while True:
            can_use = [card for card in current_cards if self._can_cast(card, mana_pool)]
            if len(can_use) == 0:
                break
            for card in can_use:
                self._cast(card, mana_pool)
                current_cards.remove(card)
                yield card, copy(mana_pool)
    
    def run(self, opening_hand):
        relevant_cards = [card for card in opening_hand if len(card.makes_mana) > 0]
        if self.mana_priority is not None:
            mana_priority_cards = []
            remaining_cards = []
            for card in relevant_cards:
                if self.mana_priority in card.mana_makes() or (Mana.ANY in card.mana_makes() and len(card.costs_mana) == 0):
                    mana_priority_cards.append(card)
                else:
                    remaining_cards.append(card)
            last_mana_pool = {}
            for card, mana_pool in self._run(mana_priority_cards, last_mana_pool):
                last_mana_pool = copy(mana_pool)
                mana_priority_cards.remove(card)
                yield card, mana_pool
            
            remaining_cards = [card for card in mana_priority_cards if not card.is_land] + remaining_cards
            yield from self._run(remaining_cards, last_mana_pool)
        else:
            yield from self._run(relevant_cards)

class NecroDeckSample:
    def __init__(self, deck):
        self.mulligans = 0
        self.hand = []
        self.initial_deck = deck
        self.deck = None
        self.played = []
        self.hand_history = []
        self.shuffle_deck()

    def in_hand(self, card_type, number_of_card):
        return (
            sum([1 for card in self.hand if isinstance(card, card_type)])
            >= number_of_card
        )

    def hand_can_produce_mana(self, to_produce, staging_hand=None):
        hand = staging_hand or self.hand
        mana_generator = ManaGenerator()
        for card, mana_pool in mana_generator.run(hand):
            if mana_costs_covered(mana_pool, to_produce):
                return True
        return False

    def hand_can_produce_mana_and_bargain(self, mana_dict):
        if not self.hand_can_produce_mana(mana_dict):
            return False

        artifact_mana_sources = [
            card
            for card in self.hand
            if len(card.makes_mana) > 0 and card.is_artifact
        ]
        non_artifact_mana_sources = [
            card
            for card in self.hand
            if len(card.makes_mana) > 0 and not card.is_artifact
        ]

        try_hand = non_artifact_mana_sources
        while (
            not self.hand_can_produce_mana(mana_dict, staging_hand=try_hand)
            and len(artifact_mana_sources) > 0
        ):
            try_hand.append(artifact_mana_sources.pop())

        return len(artifact_mana_sources) > 0

    def opening_hand_enables_necrodominance(self):
        return self.in_hand(Necrodominance, 1) and self.hand_can_produce_mana(
            {Mana.BLACK: 3}
        )

    def opening_hand_enables_beseech(self):
        return self.in_hand(Beseech, 1) and self.hand_can_produce_mana_and_bargain(
            {Mana.ANY: 1, Mana.BLACK: 3}
        )

    def opening_hand_can_attempt_win(self):
        return (
            self.opening_hand_enables_necrodominance()
            or self.opening_hand_enables_beseech()
        )

    def can_mulligan(self):
        # the next mulligan leaves us with more than 2 cards
        return 7 - self.mulligans + 1 > 2

    def get_best_initial_mana(self):
        def score_card(card):
            score = 0
            if card.is_artifact:
                score += 1
            if card.is_land:
                score += 1
            if card.mana_makes().get(Mana.BLACK, 0) + card.mana_makes().get(Mana.ANY, 0) > 0:
                score += 1
            return score
        scored = [
            (card, score_card(card))
            for card in self.hand
            if card.initial_mana
        ]
        if len(scored) == 0:
            return None
        return max(scored, key=lambda x: x[1])[0]

    def get_best_wincon(self):
        if self.in_hand(Necrodominance, 1):
            return Necrodominance()
        if self.in_hand(Beseech, 1):
            return Beseech()
        return None

    def cards_to_cover_costs(self, card_to_cover: Card):
        costs = card_to_cover.mana_costs()
        mana_priority = None
        mana_priorites = [(mana, cost) for mana, cost in costs.items() if mana != Mana.ANY]
        if len(mana_priorites) > 0:
            mana_priorites = sorted(mana_priorites, key=lambda x: x[1], reverse=True)
            mana_priority = mana_priorites[0][0]
        mana_generator = ManaGenerator(mana_priority=mana_priority)

        cards_to_cover_costs = []
        for card, mana_pool in mana_generator.run(self.hand):
            cards_to_cover_costs.append(card)
            if mana_costs_covered(mana_pool, costs):
                return cards_to_cover_costs
        
        return []


    def select_red_acceleration_and_fixing(self, cards_to_pick_from, number_of_cards):
        acceleration_cards = [
            card
            for card in cards_to_pick_from
            if card.amount_of_mana_produced(Mana.RED) > 0 and not card.is_land
        ]
        acceleration_cards = sorted(
            acceleration_cards,
            key=lambda x: x.amount_of_mana_produced(Mana.RED),
            reverse=True,
        )
        cards_chosen = acceleration_cards[:number_of_cards]
        if len(cards_chosen) < number_of_cards:
            fixing_cards = [card for card in cards_to_pick_from if card.fixes]
            cards_chosen += fixing_cards[: number_of_cards - len(cards_chosen)]
        return cards_chosen

    def select_remaining_cards(self, cards_to_pick_from, number_of_cards):
        protection = [card for card in cards_to_pick_from if card.is_protection]
        protection = protection[:number_of_cards]
        if len(protection) < number_of_cards:
            return protection + self.select_red_acceleration_and_fixing(
                cards_to_pick_from,
                number_of_cards - len(protection)
            )
        return protection

    def keep_best_cards(self, cards_we_can_keep):
        cards_kept = 0
        initial_mana = self.get_best_initial_mana()
        wincon = None
        acceleration_cards = []
        rest_of_cards = []
        if initial_mana is not None:
            cards_kept += 1
            wincon = self.get_best_wincon()
        if wincon is not None and cards_kept < cards_we_can_keep:
            cards_kept += 1
            acceleration_cards = self.cards_to_cover_costs(wincon)
            if initial_mana in acceleration_cards:
                acceleration_cards.remove(initial_mana)
            acceleration_cards = acceleration_cards[: cards_we_can_keep - cards_kept]
            cards_kept += len(acceleration_cards)
        if len(acceleration_cards) > 0 and cards_kept < cards_we_can_keep:
            remaining_cards = cards_we_can_keep - cards_kept
            picked_cards = [initial_mana] + acceleration_cards + [wincon]
            cards_to_pick_from = [card for card in self.hand if card not in picked_cards]
            rest_of_cards = self.select_remaining_cards(cards_to_pick_from, remaining_cards)

        self.hand = [initial_mana, wincon] + acceleration_cards + rest_of_cards
        self.hand = [card for card in self.hand if card is not None]

    def shuffle_deck(self):
        self.deck = copy(self.initial_deck)
        random.shuffle(self.deck)

    def handle_mulligan(self):
        self.shuffle_deck()

        cards_we_can_keep = 7 - self.mulligans
        cards_available = 7

        self.hand = []
        while cards_available > 0:
            self.hand.append(self.deck.pop())
            cards_available -= 1
        
        self.hand_history.append(copy(self.hand))
        self.mulligans += 1

        self.keep_best_cards(cards_we_can_keep)

    def sample_result(self):
        return SampleResult(
            protected=any([card.is_protection for card in self.hand]),
            won=self.opening_hand_can_attempt_win(),
            hand_history=self.hand_history
        )

    def sample_wins(self) -> SampleResult:
        while not self.opening_hand_can_attempt_win() and self.can_mulligan():
            self.handle_mulligan()

        self.hand_history.append(copy(self.hand))
        return self.sample_result()


tony_decklist = [
    GemstoneMine(),
    GemstoneMine(),
    GemstoneMine(),
    Vault(),
    Vault(),
    Vault(),
    Vault(),
    Petal(),
    Petal(),
    Petal(),
    Petal(),
    PactOfNegation(),
    PactOfNegation(),
    PactOfNegation(),
    PactOfNegation(),
    SummonersPact(),
    SummonersPact(),
    SummonersPact(),
    SummonersPact(),
    DarkRitual(),
    DarkRitual(),
    DarkRitual(),
    DarkRitual(),
    WildCantor(),
    BorneOnAWind(),
    BorneOnAWind(),
    BorneOnAWind(),
    CabalRitual(),
    CabalRitual(),
    CabalRitual(),
    CabalRitual(),
    Manamorphose(),
    Manamorphose(),
    Manamorphose(),
    Manamorphose(),
    Necrodominance(),
    Necrodominance(),
    Necrodominance(),
    Necrodominance(),
    SimianSpiritGuide(),
    SimianSpiritGuide(),
    SimianSpiritGuide(),
    SimianSpiritGuide(),
    ValakutAwakening(),
    ValakutAwakening(),
    ValakutAwakening(),
    ValakutAwakening(),
    ElvishSpiritGuide(),
    ElvishSpiritGuide(),
    ElvishSpiritGuide(),
    ElvishSpiritGuide(),
    Tendrils(),
    Beseech(),
    Beseech(),
    Beseech(),
    Beseech(),
    Chancellor(),
    Chancellor(),
    Chancellor(),
    Chancellor(),
]
original_tony = copy(tony_decklist)


n_trials = 10000

results = []

for i in range(n_trials):
    print(f"running sample n {i}/{n_trials}")
    deck = NecroDeckSample(tony_decklist)
    results.append(deck.sample_wins())


won_results = [result for result in results if result.won]
for result in results:
    print(result)
    print("\n")

necro_hit_rate = len(won_results) / n_trials
print(f"necro hit rate: {necro_hit_rate}")


def get_deck_post_win(result):
    original_deck = original_tony[::]
    last_hand = result.hand_history[-1]
    new_hand = last_hand[::]
    cards_to_remove = [card for card in last_hand if isinstance(card, Necrodominance) or len(card.makes_mana) > 0]
    beseeches = len([card for card in last_hand if isinstance(card, Beseech)])
    if beseeches > 0:
        remove_beseeches = [card for card in last_hand if isinstance(card, Beseech)]
        cards_to_remove += remove_beseeches[1:]
    summoners_pacts = [card for card in last_hand if isinstance(card, SummonersPact)]
    guides_remaining = len([card for card in original_deck if isinstance(card, ElvishSpiritGuide) ]) - len([card for card in last_hand if isinstance(card, ElvishSpiritGuide)])
    for pact in summoners_pacts:
        if guides_remaining > 0:
            original_deck.remove(ElvishSpiritGuide())
            guides_remaining -= 1
        else:
            original_deck.remove(SimianSpiritGuide())



    for card in cards_to_remove:
        deck_dict = defaultdict(int)
        for deck_print in original_deck:
            deck_dict[deck_print.__class__.__name__] += 1
        original_deck.remove(card)
        new_hand.remove(card)
    
    return original_deck, new_hand


print("running post necro simulation")
post_win_decks_and_hands = [get_deck_post_win(result) for result in won_results]

@dataclass
class PostNecroResult:
    won: bool
    stack: list[Card]

def estimate_post_necro_win(deck, new_hand, cards_to_draw = 19):
    random.shuffle(deck)
    necro_stack = new_hand + deck[:cards_to_draw]
    get_counts = defaultdict(int)
    for card in necro_stack:
        get_counts[card.__class__.__name__] += 1
    win_detected = True
    if (get_counts.get("Beseech", 0) + get_counts.get("Tendrils", 0)) < 1:
        win_detected = False
    if get_counts.get("Manamorphose", 0) < 1:
        win_detected = False
    guides = get_counts.get("ElvishSpiritGuide", 0) + get_counts.get("SimianSpiritGuide", 0) + get_counts.get("SummonersPact", 0)
    if guides < 2:
        win_detected = False
    if get_counts.get("BorneOnAWind", 0) < 1:
        win_detected = False
    if get_counts.get("Tendrils", 0) < 1 and (get_counts.get("DarkRitual", 0) < 1 and get_counts.get("CabalRitual", 0) < 1 and (get_counts.get("Manamorphose", 0) < 2 and guides < 4)):
        win_detected = False
    if win_detected == False:
        valakut_reattempt = get_counts.get("ValakutAwakening", 0) > 0 and ((guides >= 3 and (get_counts.get("Manamorphose", 0) > 0 or get_counts.get("SimianSpiritGuide", 0) > 0) or (get_counts.get("BorneOnAWind", 0) > 0)))
        if valakut_reattempt:
            guides_to_remove = 2
            elvish_guides_left = len([card for card in necro_stack if isinstance(card, ElvishSpiritGuide)])
            while guides_to_remove > 0:
                if elvish_guides_left > 0:
                    elvish_guides_left -= 1
                    guides_to_remove -= 1
                else:
                    guides_to_remove -= 1
            cards_that_wed_keep = {
                "SimianSpiritGuide": 1,
                "Manamorphose": 1,
                "BorneOnAWind": 1,
                "Tendrils": 1,
            }
            hand_to_keep = []
            for card in necro_stack:
                if cards_that_wed_keep.get(card.__class__.__name__, 0) > 0:
                    cards_that_wed_keep[card.__class__.__name__] -= 1
                    hand_to_keep.append(card)
                else:
                    if card in deck:
                        deck.remove(card)
            return estimate_post_necro_win(deck, hand_to_keep, 19 - len(hand_to_keep))
    return PostNecroResult(won=win_detected, stack=necro_stack)

print("estimating post necro win percentage")
post_necro_results = [estimate_post_necro_win(deck, hand) for deck, hand in post_win_decks_and_hands]
post_necro_percentage = len([result for result in post_necro_results if result.won]) / len(post_necro_results)
print(f"post necro win percentage: {round(post_necro_percentage * 100, 2)}%")

print(f"Expected T1 wins with necrodominance, without accounting for protection/interaction - {necro_hit_rate} * {post_necro_percentage} = {round(post_necro_percentage * necro_hit_rate * 100, 2)}%")

