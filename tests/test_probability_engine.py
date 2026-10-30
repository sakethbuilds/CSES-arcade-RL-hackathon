"""Independent mathematical unit tests for the Buckshot Roulette probability engine.
Verifies exact transition probabilities, history sensitivity, and absorbing boundary dynamics.
Independent of tournament engine.
"""
import unittest
from math import comb
import sys
import os

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bots.our_bot_v2.bot import estimate_from_history, estimate_live_probability, update_belief_step, p_live_from_belief


class TestProbabilityEngine(unittest.TestCase):
    def test_history_a_and_b_exact(self):
        """Test the exact case: Initial L=1, B=2
        History A: U then B -> P(Live) must be exactly 1/3
        History B: B then U -> P(Live) must be exactly 1/2
        """
        # Step-by-step verification for History A:
        # Step 0: Initial chamber (1, 2)
        b0 = {(1, 2): 1.0}
        
        # Step 1: Ejection of Unknown shell (U)
        # In (1, 2), generator flipped fair coin (both L and B > 0).
        # With prob 0.5, ejected Live -> chamber becomes (0, 2).
        # With prob 0.5, ejected Blank -> chamber becomes (1, 1).
        b1_a = update_belief_step(b0, 'U')
        self.assertAlmostEqual(b1_a.get((0, 2), 0.0), 0.5, places=9)
        self.assertAlmostEqual(b1_a.get((1, 1), 0.0), 0.5, places=9)

        # Step 2: Consumption of known Blank shell (B)
        # From (0, 2), next shell is deterministically Blank (prob 1.0). Chamber becomes (0, 1). Weight = 0.5 * 1.0 = 0.50.
        # From (1, 1), next shell is Blank with prob 0.5. Chamber becomes (1, 0). Weight = 0.5 * 0.5 = 0.25.
        # Total evidence = 0.50 + 0.25 = 0.75.
        # Normalized posterior: P((0, 1)) = 0.50/0.75 = 2/3. P((1, 0)) = 0.25/0.75 = 1/3.
        b2_a = update_belief_step(b1_a, 'B')
        self.assertAlmostEqual(b2_a.get((0, 1), 0.0), 2.0 / 3.0, places=9)
        self.assertAlmostEqual(b2_a.get((1, 0), 0.0), 1.0 / 3.0, places=9)

        # Next shell Live probability:
        # Under (0, 1): 0 Live remain -> P(Live) = 0.0.
        # Under (1, 0): 1 Live, 0 Blank remain -> P(Live) = 1.0.
        # E[P(Live)] = (2/3)*0 + (1/3)*1.0 = 1/3.
        p_live_a = p_live_from_belief(b2_a)
        self.assertAlmostEqual(p_live_a, 1.0 / 3.0, places=9)

        # Combined estimate_from_history call
        p_a_calc = estimate_from_history(1, 2, ['U', 'B'])
        self.assertAlmostEqual(p_a_calc, 1.0 / 3.0, places=9)

        # -------------------------------------------------------------
        # Step-by-step verification for History B:
        # Step 1: Consumption of known Blank shell (B)
        # In (1, 2), next shell was Blank (prob 0.5).
        # Chamber transitions deterministically to (1, 1).
        b1_b = update_belief_step(b0, 'B')
        self.assertAlmostEqual(b1_b.get((1, 1), 0.0), 1.0, places=9)

        # Step 2: Ejection of Unknown shell (U)
        # From (1, 1), both L=1 and B=1 remain. Fair coin flip (prob 0.5 each).
        # If Live: chamber becomes (0, 1), prob 0.5.
        # If Blank: chamber becomes (1, 0), prob 0.5.
        b2_b = update_belief_step(b1_b, 'U')
        self.assertAlmostEqual(b2_b.get((0, 1), 0.0), 0.5, places=9)
        self.assertAlmostEqual(b2_b.get((1, 0), 0.0), 0.5, places=9)

        # Next shell Live probability:
        # Under (0, 1): 0 Live remain -> P(Live) = 0.0.
        # Under (1, 0): 1 Live, 0 Blank remain -> P(Live) = 1.0.
        # E[P(Live)] = 0.5*0 + 0.5*1.0 = 1/2.
        p_live_b = p_live_from_belief(b2_b)
        self.assertAlmostEqual(p_live_b, 0.5, places=9)

        # Combined estimate_from_history call
        p_b_calc = estimate_from_history(1, 2, ['B', 'U'])
        self.assertAlmostEqual(p_b_calc, 0.5, places=9)

    def test_non_permutation_invariance_proof(self):
        """Mathematical proof that History A and History B are distinct."""
        p_a = estimate_from_history(1, 2, ['U', 'B'])
        p_b = estimate_from_history(1, 2, ['B', 'U'])
        self.assertNotEqual(p_a, p_b)
        self.assertAlmostEqual(p_a, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(p_b, 1.0 / 2.0, places=9)

    def test_exhaustive_against_path_enumeration(self):
        """Verify sequential belief updates against brute-force path enumeration."""
        def enumerate_paths(l_init, b_init):
            paths = {}
            def dfs(rem_l, rem_b, path, prob):
                if len(path) == l_init + b_init:
                    paths[tuple(path)] = prob
                    return
                if rem_l > 0 and rem_b > 0:
                    dfs(rem_l - 1, rem_b, path + [True], prob * 0.5)
                    dfs(rem_l, rem_b - 1, path + [False], prob * 0.5)
                elif rem_l > 0:
                    dfs(rem_l - 1, rem_b, path + [True], prob * 1.0)
                else:
                    dfs(rem_l, rem_b - 1, path + [False], prob * 1.0)
            dfs(l_init, b_init, [], 1.0)
            return paths

        # Test across multiple small chamber configurations
        for L, B, hist in [
            (1, 2, ['U', 'B']),
            (1, 2, ['B', 'U']),
            (1, 3, ['U', 'B', 'B']),
            (1, 3, ['B', 'U', 'B']),
            (1, 3, ['B', 'B', 'U']),
            (2, 2, ['U', 'L']),
            (2, 2, ['L', 'U']),
            (2, 3, ['U', 'B', 'L']),
        ]:
            paths = enumerate_paths(L, B)
            total_matching_weight = 0.0
            live_next_weight = 0.0
            k = len(hist)

            for path, prob in paths.items():
                match = True
                for i, h in enumerate(hist):
                    if h == 'L' and not path[i]: match = False; break
                    elif h == 'B' and path[i]: match = False; break
                    elif h == 'U': pass # any matches U
                if match:
                    total_matching_weight += prob
                    if k < len(path) and path[k]:
                        live_next_weight += prob

            ground_truth_p = live_next_weight / total_matching_weight if total_matching_weight > 0 else 0.0
            engine_p = estimate_from_history(L, B, hist)
            self.assertAlmostEqual(engine_p, ground_truth_p, places=8,
                                   msg=f"Failed for L={L}, B={B}, hist={hist}")


if __name__ == "__main__":
    unittest.main()

