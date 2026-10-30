"""Bot v2: Survival-First Bayesian Expectimax Agent for Buckshot Roulette.
Optimized for 4-Player Multiplayer and 1v1 Endgames.
CSES Arcade RL Hackathon.
"""
from math import comb
from typing import Dict, List, Optional, Tuple

from schemas import (
    Action_PICK_UPGRADES,
    Action_PLAY,
    ActionType,
    NextRound,
    Obs_PICK_UPGRADES,
    Obs_PLAY,
    UpgradeType,
)


def update_belief_step(belief: Dict[Tuple[int, int], float], event: str) -> Dict[Tuple[int, int], float]:
    """Applies one observation event ('L', 'B', or 'U') to the chamber belief distribution."""
    new_belief: Dict[Tuple[int, int], float] = {}
    for (l, b), prob in belief.items():
        if prob <= 0:
            continue
        if l > 0 and b > 0:
            p_live_step = 0.5
            p_blank_step = 0.5
        elif l > 0:
            p_live_step = 1.0
            p_blank_step = 0.0
        else:
            p_live_step = 0.0
            p_blank_step = 1.0

        if event == 'L':
            if p_live_step > 0 and l > 0:
                new_belief[(l - 1, b)] = new_belief.get((l - 1, b), 0.0) + prob * p_live_step
        elif event == 'B':
            if p_blank_step > 0 and b > 0:
                new_belief[(l, b - 1)] = new_belief.get((l, b - 1), 0.0) + prob * p_blank_step
        elif event == 'U':
            if p_live_step > 0 and l > 0:
                new_belief[(l - 1, b)] = new_belief.get((l - 1, b), 0.0) + prob * p_live_step
            if p_blank_step > 0 and b > 0:
                new_belief[(l, b - 1)] = new_belief.get((l, b - 1), 0.0) + prob * p_blank_step

    total = sum(new_belief.values())
    return {k: v / total for k, v in new_belief.items()} if total > 0 else {}


def p_live_from_belief(belief: Dict[Tuple[int, int], float]) -> float:
    """Computes expected live probability of the next shell given current belief distribution."""
    p = 0.0
    for (l, b), prob in belief.items():
        if l > 0 and b > 0:
            p += prob * 0.5
        elif l > 0:
            p += prob * 1.0
        else:
            p += 0.0
    return p


def estimate_from_history(l_init: int, b_init: int, history: List[str]) -> float:
    """Exact sequential belief estimation when chronological event history is known.

    Guarantees:
    - History A (L=1, B=2, ['U', 'B']): returns exactly 1/3 (0.333333...)
    - History B (L=1, B=2, ['B', 'U']): returns exactly 1/2 (0.500000...)
    """
    belief = {(l_init, b_init): 1.0}
    for e in history:
        belief = update_belief_step(belief, e)
    return p_live_from_belief(belief)


def estimate_live_probability(obs: Obs_PLAY, history: Optional[List[str]] = None) -> float:
    """Exact belief-state dynamic programming probability estimator.

    Respects the sequential shell generation algorithm with absorbing boundaries
    in game_API._setup_round(). When chronological history is known, evaluates
    exact sequential updates. When only aggregate counts are available, rigorously
    marginalizes over possible event orderings without false permutation assumptions.
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
    m_consumed = l_known + b_known + u

    # Definite bounds
    if l_known >= l_init:
        return 0.0
    if b_known >= b_init:
        return 1.0

    # If exact chronological history is available, evaluate sequential transitions directly:
    if history is not None and len(history) == m_consumed:
        return estimate_from_history(l_init, b_init, history)

    # If no unknown ejections exist, remaining chamber counts are exact:
    if u == 0:
        l_gun = l_init - l_known
        b_gun = b_init - b_known
        if l_gun <= 0:
            return 0.0
        if b_gun <= 0:
            return 1.0
        return 0.5

    # If unknown ejections occurred and order is ambiguous, marginalize over valid paths:
    dp: Dict[int, float] = {0: 1.0}
    for m in range(m_consumed):
        new_dp: Dict[int, float] = {}
        for l_used, prob in dp.items():
            if prob <= 0:
                continue
            b_used = m - l_used
            rem_l = l_init - l_used
            rem_b = b_init - b_used
            if rem_l > 0 and rem_b > 0:
                new_dp[l_used + 1] = new_dp.get(l_used + 1, 0.0) + prob * 0.5
                new_dp[l_used] = new_dp.get(l_used, 0.0) + prob * 0.5
            elif rem_l > 0:
                new_dp[l_used + 1] = new_dp.get(l_used + 1, 0.0) + prob * 1.0
            else:
                new_dp[l_used] = new_dp.get(l_used, 0.0) + prob * 1.0
        dp = new_dp

    k_min = max(0, u - (b_init - b_known))
    k_max = min(u, l_init - l_known)
    weights: Dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        l_used = l_known + k
        b_used = b_known + (u - k)
        p_path = dp.get(l_used, 0.0)
        w = p_path * comb(l_used, l_known) * comb(b_used, b_known)
        if w > 0:
            weights[k] = w

    total_w = sum(weights.values())
    if total_w <= 0:
        return max(0.0, min(1.0, (l_init - l_known) / max(1, r)))

    expected_prob = 0.0
    for k, w in weights.items():
        rem_l = l_init - l_known - k
        rem_b = b_init - b_known - (u - k)
        if rem_l > 0 and rem_b > 0:
            p_k = 0.5
        elif rem_l > 0:
            p_k = 1.0
        else:
            p_k = 0.0
        expected_prob += (w / total_w) * p_k

    return expected_prob


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
        self.params: dict = {
            "self_shot_p_threshold": 0.20,
            "threat_weight_saw": 25.0,
            "threat_weight_mag": 15.0,
            "threat_weight_float": 10.0,
            "threat_weight_juice": 5.0,
            "leader_pressure_weight": 8.0,
            "lethal_base_bonus": 100.0,
            "lethal_threat_multiplier": 5.0,
            "next_player_danger_weight": 10.0,
            "saw_cost_penalty": 20.0,
        }
        if params:
            self.params.update(params)
        self.saws_used_this_turn = 0
        self.last_rounds_seen = -1
        self.last_session_seen = -1

    def init(self, params: Optional[dict] = None) -> None:
        if params:
            self.params.update(params)
        self.saws_used_this_turn = 0
        self.last_rounds_seen = -1
        self.last_session_seen = -1

    def _sync_turn_state(self, obs: Obs_PLAY) -> None:
        """Stateless synchronization: resets turn trackers whenever chamber moves."""
        if obs.session_no != self.last_session_seen or obs.rounds_remaining != self.last_rounds_seen:
            self.saws_used_this_turn = 0
            self.last_session_seen = obs.session_no
            self.last_rounds_seen = obs.rounds_remaining

    def _compute_threat(self, p) -> float:
        """Calculates offensive threat level of a player's upgrade arsenal."""
        u = p.upgrades
        return (
            u.hand_saw * self.params.get("threat_weight_saw", 25.0)
            + u.mag_glass * self.params.get("threat_weight_mag", 15.0)
            + u.pool_floatie * self.params.get("threat_weight_float", 10.0)
            + u.juice * self.params.get("threat_weight_juice", 5.0)
        )

    def _evaluate_target_and_saws(self, obs: Obs_PLAY, living_opponents: List[int]) -> Tuple[int, int]:
        """Jointly evaluates optimal target and Hand Saw allocation."""
        me = obs.turn
        my_saws = obs.players[me].upgrades.hand_saw
        total_saws_available = my_saws + self.saws_used_this_turn

        if not living_opponents:
            return (me, 0)
        if len(living_opponents) == 1:
            opp = living_opponents[0]
            opp_hp = obs.players[opp].health
            # In 1v1: use all saws needed to achieve lethal or max damage
            saws = min(total_saws_available, max(0, opp_hp - 1))
            return (opp, saws)

        # In multiplayer:
        min_opp_hp = min(obs.players[i].health for i in living_opponents)
        immediate_next = get_next_turn(obs, me)

        best_score = -1e9
        best_target = living_opponents[0]
        best_saws = 0

        for i in living_opponents:
            p = obs.players[i]
            threat = self._compute_threat(p)
            saws_needed = max(0, p.health - 1)

            # Option A: Guaranteed lethal elimination if we have enough saws
            if total_saws_available >= saws_needed:
                elim_score = (
                    self.params.get("lethal_base_bonus", 100.0)
                    + threat * self.params.get("lethal_threat_multiplier", 5.0)
                    - saws_needed * self.params.get("saw_cost_penalty", 20.0)
                )
                if elim_score > best_score:
                    best_score = elim_score
                    best_target = i
                    best_saws = saws_needed

            # Option B: Non-lethal shot (0 saws)
            leader_diff = p.health - min_opp_hp
            non_lethal_score = (
                threat
                + leader_diff * self.params.get("leader_pressure_weight", 8.0)
            )
            if i == immediate_next and not p.has_pool_floatie_on and threat > 0:
                non_lethal_score += self.params.get("next_player_danger_weight", 10.0)

            if non_lethal_score > best_score:
                best_score = non_lethal_score
                best_target = i
                best_saws = 0

        return (best_target, best_saws)

    def play(self, obs: Obs_PLAY) -> Action_PLAY:
        self._sync_turn_state(obs)
        me = obs.turn
        my_player = obs.players[me]
        my_upgrades = my_player.upgrades
        my_health = my_player.health
        r = obs.rounds_remaining

        living_opponents = [i for i, p in enumerate(obs.players) if i != me and p.health > 0]
        alive_count = len(living_opponents) + 1
        current_damage = 1 + self.saws_used_this_turn

        p_live = estimate_live_probability(obs)

        # -------------------------------------------------------------
        # STEP 1: INFORMATION GATHERING (MAGNIFYING GLASS)
        # -------------------------------------------------------------
        if obs.next_round == NextRound.UNKNOWN and r > 0 and my_upgrades.mag_glass > 0:
            if 0.0 < p_live < 1.0:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.MAG_GLASS
                action.target = None
                return action

        # Re-evaluate live probability
        p_live = estimate_live_probability(obs)

        # -------------------------------------------------------------
        # STEP 2: CONFIRMED BLANK -> SHOOT SELF (FREE TURN)
        # -------------------------------------------------------------
        if obs.next_round == NextRound.BLANK or p_live == 0.0:
            action = Action_PLAY()
            action.action = ActionType.SHOOT
            action.useUpgrade = None
            action.target = me
            self.saws_used_this_turn = 0
            return action

        # -------------------------------------------------------------
        # STEP 3: CONFIRMED LIVE -> OFFENSIVE MULTIPLIERS & SHOOT
        # -------------------------------------------------------------
        if obs.next_round == NextRound.LIVE or p_live == 1.0:
            best_opp, target_saws = self._evaluate_target_and_saws(obs, living_opponents)

            # 3A. Apply Hand Saws as planned
            if self.saws_used_this_turn < target_saws and my_upgrades.hand_saw > 0:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.HAND_SAW
                action.target = None
                self.saws_used_this_turn += 1
                return action

            target_hp = obs.players[best_opp].health if living_opponents else 0
            will_target_die = (target_hp <= current_damage)

            # 3B. Pool Floatie Turn Denial
            if my_upgrades.pool_floatie > 0 and living_opponents:
                float_target = None
                if alive_count == 2:
                    # In 1v1: float opponent if they will SURVIVE
                    if not will_target_die and not obs.players[best_opp].has_pool_floatie_on:
                        float_target = best_opp
                else:
                    # In multiplayer: float next active player if they threaten us with lethal saws
                    next_player = get_next_turn(obs, me)
                    if next_player != me and next_player != best_opp:
                        p_next = obs.players[next_player]
                        if not p_next.has_pool_floatie_on and p_next.upgrades.hand_saw >= 1 and my_health <= 3:
                            float_target = next_player

                if float_target is not None:
                    action = Action_PLAY()
                    action.action = ActionType.USE_UPGRADE
                    action.useUpgrade = UpgradeType.POOL_FLOATIE
                    action.target = float_target
                    return action

            # 3C. Juice (Heal before passing turn to opponents)
            if my_health < obs.max_health and my_upgrades.juice > 0:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.JUICE
                action.target = None
                return action

            # 3D. Execute Confirmed Live Shot
            action = Action_PLAY()
            action.action = ActionType.SHOOT
            action.useUpgrade = None
            action.target = best_opp if living_opponents else me
            self.saws_used_this_turn = 0
            return action

        # -------------------------------------------------------------
        # STEP 4: UNCERTAIN SHELL (0.0 < p_live < 1.0)
        # -------------------------------------------------------------
        # 4A. Defensive Coffee Usage
        if my_upgrades.coffee > 0 and r > 0 and obs.next_round == NextRound.UNKNOWN:
            if r == 1:
                if my_health <= 1 and 0.35 <= p_live <= 0.65:
                    action = Action_PLAY()
                    action.action = ActionType.USE_UPGRADE
                    action.useUpgrade = UpgradeType.COFFEE
                    action.target = None
                    return action
            else:
                if my_health <= 2 and p_live >= 0.40:
                    action = Action_PLAY()
                    action.action = ActionType.USE_UPGRADE
                    action.useUpgrade = UpgradeType.COFFEE
                    action.target = None
                    return action

        # 4B. Juice before uncertain action
        if my_health < obs.max_health and my_upgrades.juice > 0:
            action = Action_PLAY()
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.JUICE
            action.target = None
            return action

        # 4B2. In 1v1, use Pool Floatie before shooting opponent under uncertainty
        if alive_count == 2 and my_upgrades.pool_floatie > 0 and living_opponents:
            opp = living_opponents[0]
            if not obs.players[opp].has_pool_floatie_on:
                action = Action_PLAY()
                action.action = ActionType.USE_UPGRADE
                action.useUpgrade = UpgradeType.POOL_FLOATIE
                action.target = opp
                return action

        # 4C. Shooting Decision under Uncertainty
        self_shot_thresh = self.params.get("self_shot_p_threshold", 0.20)
        action = Action_PLAY()
        action.action = ActionType.SHOOT
        action.useUpgrade = None

        if p_live < self_shot_thresh and my_health > 1:
            action.target = me
        else:
            best_opp, _ = self._evaluate_target_and_saws(obs, living_opponents)
            action.target = best_opp if living_opponents else me

        self.saws_used_this_turn = 0
        return action

    def pick_upgrade(self, obs: Obs_PICK_UPGRADES) -> Action_PICK_UPGRADES:
        """Contextual upgrade drafting with multiplayer survival focus."""
        action = Action_PICK_UPGRADES()
        me = obs.turn
        my_player = obs.players[me]
        my_health = my_player.health
        avail = obs.available_upgrades
        living_players = sum(1 for p in obs.players if p.health > 0)

        priority: List[UpgradeType] = []

        # Urgent defense: Juice is critical when low
        if my_health <= 2 and avail.juice > 0:
            priority.append(UpgradeType.JUICE)

        # 1v1 Duel Mode: Pool Floatie is S-Tier (double turn steal)
        if living_players == 2 and obs.session_no >= 1 and avail.pool_floatie > 0:
            priority.append(UpgradeType.POOL_FLOATIE)

        # In multiplayer (>= 3 players), survival rules: JUICE IS KING
        if living_players >= 3:
            priority.extend([
                UpgradeType.JUICE,
                UpgradeType.MAG_GLASS,
                UpgradeType.HAND_SAW,
                UpgradeType.POOL_FLOATIE,
                UpgradeType.COFFEE,
            ])
        else:
            # In 1v1, offense & turn steal rule
            priority.extend([
                UpgradeType.MAG_GLASS,
                UpgradeType.HAND_SAW,
                UpgradeType.POOL_FLOATIE,
                UpgradeType.JUICE,
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
