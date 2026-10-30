"""Competitive Bayesian Expectimax Bot for Buckshot Roulette.
CSES Arcade RL Hackathon.
"""
from typing import List, Optional
from schemas import (
    Action_PICK_UPGRADES,
    Action_PLAY,
    ActionType,
    NextRound,
    Obs_PICK_UPGRADES,
    Obs_PLAY,
    UpgradeType,
)


def estimate_live_probability(obs: Obs_PLAY) -> float:
    """Estimates the probability that the next shell in the gun chamber is LIVE.

    Accounts for:
    - Known next shell (from Magnifying Glass)
    - Initial live / blank counts
    - Cumulative fired shells
    - Cumulative known ejections (own Coffee)
    - Cumulative unknown ejections (opponents' Coffee)
    - Non-uniform generation mechanics in game_API._setup_round()
    """
    if obs.next_round == NextRound.LIVE:
        return 1.0
    if obs.next_round == NextRound.BLANK:
        return 0.0

    r = obs.rounds_remaining
    if r <= 0:
        return 0.0

    l_init = obs.lives_init
    b_init = obs.blanks_init

    l_known = obs.lives_shot + obs.ejections.lives_ejected
    b_known = obs.blanks_shot + obs.ejections.blanks_ejected
    u = obs.ejections.unknowns_ejected

    # Definite bounds
    if l_known >= l_init:
        return 0.0
    if b_known >= b_init:
        return 1.0

    # If no unknown ejections, remaining counts in chamber are exact:
    if u == 0:
        l_gun = l_init - l_known
        b_gun = b_init - b_known
        if l_gun <= 0:
            return 0.0
        if b_gun <= 0:
            return 1.0
        # Simulator mechanics: as long as both live and blank shells remain,
        # each step was chosen via rng.randrange(2), giving exactly 50% probability!
        return 0.5

    # If there are unknown ejections (opponents drank coffee):
    # k is the number of live shells among the u unknown ejections.
    k_min = max(0, u - (b_init - b_known))
    k_max = min(u, l_init - l_known)

    if k_min > k_max:
        return max(0.0, min(1.0, (l_init - l_known) / max(1, r)))

    # Marginalize over possible k using binomial weights for the unknown ejections
    total_weight = 0.0
    expected_prob = 0.0

    for k in range(k_min, k_max + 1):
        from math import comb
        weight = comb(u, k)
        l_gun = l_init - l_known - k
        b_gun = b_init - b_known - (u - k)

        if l_gun <= 0:
            prob_k = 0.0
        elif b_gun <= 0:
            prob_k = 1.0
        else:
            prob_k = 0.5

        expected_prob += weight * prob_k
        total_weight += weight

    return expected_prob / total_weight if total_weight > 0 else 0.5


def get_next_turn(obs: Obs_PLAY, from_turn: int) -> int:
    """Computes which player would take the turn next, considering floaties and health."""
    player_count = len(obs.players)
    for offset in range(1, player_count + 1):
        candidate = (from_turn + offset) % player_count
        if obs.players[candidate].health <= 0:
            continue
        if obs.players[candidate].has_pool_floatie_on:
            continue
        return candidate
    return from_turn


class Bot:
    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or {}
        self.current_damage = 1
        self.last_turn_seen = -1
        self.last_session_seen = -1
        self.consecutive_self_shots = 0

    def init(self, params: Optional[dict] = None) -> None:
        if params:
            self.params.update(params)
        self.current_damage = 1
        self.last_turn_seen = -1
        self.last_session_seen = -1
        self.consecutive_self_shots = 0

    def _sync_state(self, obs: Obs_PLAY) -> None:
        """Reset internal damage and shot tracking when session or active turn switches."""
        if obs.session_no != self.last_session_seen or obs.turn != self.last_turn_seen:
            self.current_damage = 1
            self.consecutive_self_shots = 0
            self.last_session_seen = obs.session_no
            self.last_turn_seen = obs.turn

    def play(self, obs: Obs_PLAY) -> Action_PLAY:
        self._sync_state(obs)
        me = obs.turn
        my_player = obs.players[me]
        my_upgrades = my_player.upgrades
        my_health = my_player.health
        r = obs.rounds_remaining

        living_opponents = [i for i, p in enumerate(obs.players) if i != me and p.health > 0]

        # -------------------------------------------------------------
        # STEP A: HEALING (JUICE)
        # -------------------------------------------------------------
        if my_health < obs.max_health and my_upgrades.juice > 0:
            action = Action_PLAY()
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.JUICE
            action.target = None
            return action

        # -------------------------------------------------------------
        # STEP B: INFORMATION GATHERING (MAGNIFYING GLASS)
        # -------------------------------------------------------------
        p_live = estimate_live_probability(obs)
        if obs.next_round == NextRound.UNKNOWN and r > 0 and my_upgrades.mag_glass > 0:
            if 0.0 < p_live < 1.0:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.MAG_GLASS
                action.target = None
                return action

        # Recalculate p_live in case next_round was revealed or known
        p_live = estimate_live_probability(obs)

        # -------------------------------------------------------------
        # STEP C: TURN DENIAL (POOL FLOATIE)
        # -------------------------------------------------------------
        if my_upgrades.pool_floatie > 0:
            immediate_next = get_next_turn(obs, me)
            target_cand = None

            # Ideal target: next living opponent without floatie
            if (immediate_next != me and
                    obs.players[immediate_next].health > 0 and
                    not obs.players[immediate_next].has_pool_floatie_on):
                target_cand = immediate_next
            else:
                cands = [i for i in living_opponents if not obs.players[i].has_pool_floatie_on]
                if cands:
                    cands.sort(
                        key=lambda i: (
                            obs.players[i].upgrades.hand_saw,
                            obs.players[i].upgrades.mag_glass,
                            obs.players[i].health,
                        ),
                        reverse=True,
                    )
                    target_cand = cands[0]

            if target_cand is not None:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.POOL_FLOATIE
                action.target = target_cand
                return action

        # -------------------------------------------------------------
        # STEP D: DAMAGE MULTIPLICATION (HAND SAW)
        # -------------------------------------------------------------
        if p_live == 1.0 and my_upgrades.hand_saw > 0 and living_opponents:
            best_opp = self._select_best_target(obs, living_opponents, self.current_damage)
            target_hp = obs.players[best_opp].health

            if target_hp > self.current_damage:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.HAND_SAW
                action.target = None
                self.current_damage += 1
                return action

        # -------------------------------------------------------------
        # STEP E: COFFEE (DEFENSIVE / PROBING)
        # -------------------------------------------------------------
        if (my_upgrades.coffee > 0 and r > 0 and
                obs.next_round == NextRound.UNKNOWN and
                my_upgrades.mag_glass == 0):
            if r == 1 or (my_health <= 1 and p_live >= 0.5):
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.COFFEE
                action.target = None
                return action

        # -------------------------------------------------------------
        # STEP F: SHOOTING DECISION
        # -------------------------------------------------------------
        action = Action_PLAY()
        action.action = ActionType.SHOOT
        action.useUpgrade = None

        # Case 1: Confirmed BLANK -> Shoot OURSELF to keep turn!
        if obs.next_round == NextRound.BLANK or p_live == 0.0:
            if self.consecutive_self_shots < 10:
                action.target = me
                self.consecutive_self_shots += 1
                self.current_damage = 1
                return action

        # Case 2: Confirmed LIVE -> Shoot best OPPONENT!
        if obs.next_round == NextRound.LIVE or p_live == 1.0:
            best_opp = self._select_best_target(obs, living_opponents, self.current_damage)
            action.target = best_opp
            self.current_damage = 1
            self.consecutive_self_shots = 0
            return action

        # Case 3: Uncertain shell (p_live around 0.50)
        # Shooting opponent avoids 50% self-damage risk.
        if p_live < 0.20 and my_health > 1:
            action.target = me
            self.consecutive_self_shots += 1
        else:
            best_opp = self._select_best_target(obs, living_opponents, self.current_damage)
            action.target = best_opp
            self.consecutive_self_shots = 0

        self.current_damage = 1
        return action

    def _select_best_target(self, obs: Obs_PLAY, living_opponents: List[int], damage: int) -> int:
        """Heuristic evaluation for target selection.

        Priorities:
        1. Immediate lethal elimination (+1000).
        2. Threat score (holding dangerous upgrades like Hand Saw, Mag Glass).
        3. Immediate next active player (only if not floated).
        4. Finishing low-health players.
        """
        if not living_opponents:
            return obs.turn

        immediate_next = get_next_turn(obs, obs.turn)

        def score(i: int) -> float:
            p = obs.players[i]
            s = 0.0

            # Priority 1: Lethal elimination
            if p.health <= damage:
                s += 1000.0
                s += p.upgrades.hand_saw * 30.0 + p.upgrades.mag_glass * 20.0 + p.upgrades.pool_floatie * 15.0
                return s

            # Priority 2: Weaponized arsenal threat
            s += p.upgrades.hand_saw * 25.0
            s += p.upgrades.mag_glass * 15.0
            s += p.upgrades.pool_floatie * 10.0

            # Priority 3: Immediate next active opponent (only if NOT currently floated)
            if i == immediate_next and not p.has_pool_floatie_on:
                s += 15.0

            # Priority 4: Low health preference (bring closer to elimination)
            s += (obs.max_health - p.health) * 4.0

            return s

        return max(living_opponents, key=score)

    def pick_upgrade(self, obs: Obs_PICK_UPGRADES) -> Action_PICK_UPGRADES:
        """Contextual upgrade drafting.

        Considers:
        - Information dominance (MAG_GLASS is Tier S)
        - Lethality multiplier (HAND_SAW is Tier S)
        - Critical survival (JUICE when HP <= 2)
        - Turn denial (POOL_FLOATIE in 2-player or vs threats)
        """
        action = Action_PICK_UPGRADES()
        me = obs.turn
        my_player = obs.players[me]
        my_health = my_player.health
        avail = obs.available_upgrades
        living_players = sum(1 for p in obs.players if p.health > 0)

        # Dynamic priority list based on game state
        priority: List[UpgradeType] = []

        # Urgent defense: if health <= 2, Juice is critical to avoid being 1-shot
        if my_health <= 2 and avail.juice > 0:
            priority.append(UpgradeType.JUICE)

        # In 2-player endgame, Pool Floatie gives guaranteed turn steal
        if living_players == 2 and obs.session_no >= 1 and avail.pool_floatie > 0:
            priority.append(UpgradeType.POOL_FLOATIE)

        # High-value core offensive & informational items
        # Mag Glass and Hand Saw define game outcomes
        priority.extend([
            UpgradeType.MAG_GLASS,
            UpgradeType.HAND_SAW,
            UpgradeType.JUICE,
            UpgradeType.POOL_FLOATIE,
            UpgradeType.COFFEE,
        ])

        attr_map = {
            UpgradeType.JUICE: "juice",
            UpgradeType.COFFEE: "coffee",
            UpgradeType.MAG_GLASS: "mag_glass",
            UpgradeType.POOL_FLOATIE: "pool_floatie",
            UpgradeType.HAND_SAW: "hand_saw",
        }

        for upg in priority:
            attr = attr_map[upg]
            if getattr(avail, attr) > 0:
                action.chosen_upgrade = upg
                return action

        for upg, attr in attr_map.items():
            if getattr(avail, attr) > 0:
                action.chosen_upgrade = upg
                return action

        action.chosen_upgrade = UpgradeType.JUICE
        return action
