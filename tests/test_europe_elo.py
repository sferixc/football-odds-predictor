"""Check dated Elo initialization, result updates, and the prediction cutoff."""

from datetime import UTC, date, datetime
from types import SimpleNamespace
import unittest

import pandas as pd

from football_odds_predictor.data.europe import replay_europe_history
from football_odds_predictor.data.europe_elo import EuropeanEloSeeds
from football_odds_predictor.models.ucl import predict_ucl


def games(*fixtures):
    return pd.DataFrame([
        {
            "date": pd.Timestamp(day), "home_club_id": home, "away_club_id": away,
            "home_club_goals": score[0], "away_club_goals": score[1],
            "competition_id": "CL", "round": "Group Stage", "season": 2025,
        }
        for day, home, away, score in fixtures
    ])


class EuropeanEloTest(unittest.TestCase):
    def test_use_only_external_ratings_before_the_match_date(self):
        seeds = EuropeanEloSeeds.__new__(EuropeanEloSeeds)
        dates = list(pd.to_datetime(["2020-07-30", "2020-08-01", "2020-08-02"]))
        seeds.clubs = {1: (dates, [1550, 1620, 1750])}
        seeds.countries = {}
        seeds.league_countries = {}
        seeds.global_prior = dates, [1450, 1490, 1500]
        self.assertEqual(seeds.initial(1, dates[1]), (1550, "club", dates[0]))
        self.assertEqual(seeds.initial(2, dates[1]), (1450, "European median", dates[0]))

    def test_win_draw_loss_and_rating_conservation(self):
        # Even at equal club ratings the home team is favored by 60 Elo points.
        expected_changes = [9.948, -2.052, -14.052]
        for score, expected in zip(((1, 0), (1, 1), (0, 1)), expected_changes):
            with self.subTest(score=score):
                _, timeline = replay_europe_history(games(("2025-09-10", 1, 2, score)), {}, None)
                home = timeline[1][-1][1]
                away = timeline[2][-1][1]
                self.assertAlmostEqual(home - 1500, expected, places=2)
                self.assertAlmostEqual(home + away, 3000)

    def test_upset_moves_elo_more_than_favorite_win(self):
        seed = {1: 1700, 2: 1500}
        _, favorite_win = replay_europe_history(games(("2025-09-10", 1, 2, (1, 0))), seed, None)
        _, upset = replay_europe_history(games(("2025-09-10", 1, 2, (0, 1))), seed, None)
        self.assertGreater(upset[2][-1][1] - seed[2], favorite_win[1][-1][1] - seed[1])

    def test_same_day_results_are_excluded_and_next_day_updates(self):
        fixtures = games(
            ("2025-09-10", 1, 2, (2, 0)),
            ("2025-09-10", 1, 3, (0, 1)),
            ("2025-09-11", 1, 2, (0, 0)),
        )
        rows, timeline = replay_europe_history(fixtures, {1: 1600, 2: 1500, 3: 1400}, None)
        self.assertEqual(rows.elo_difference.iloc[:2].tolist(), [100, 200])
        self.assertEqual(rows.home_history.iloc[:2].tolist(), [0, 0])
        self.assertEqual(rows.home_history.iat[2], 2)
        self.assertAlmostEqual(rows.elo_difference.iat[2], timeline[1][0][1] - timeline[2][0][1])

    def test_capture_before_midnight_uses_earlier_history(self):
        history = ((0, 1500, "CL"),) * 5
        result_model = SimpleNamespace(predict_proba=lambda _: [[0.4, 0.3, 0.3]])
        goals_model = SimpleNamespace(predict=lambda _: [2.5])
        bundle = {
            "training_end": 2024,
            "names": {"alpha": 1, "beta": 2},
            "timeline": {
                1: [(date(2026, 11, 22), 1600, history), (date(2026, 11, 23), 1640, history)],
                2: [(date(2026, 11, 22), 1500, history)],
            },
            "competitions": {"CL": {"result": result_model, "goals": goals_model}},
            "domestic": {1, 2},
        }
        fixture = {"event_name": "Alpha·Beta"}
        kickoff = datetime(2026, 11, 24, 0, 30, tzinfo=UTC)
        early = datetime(2026, 11, 23, 22, tzinfo=UTC)
        self.assertEqual(predict_ucl(bundle, fixture, kickoff, observed_at=early)[2],
                         ["2026-11-22", "2026-11-22"])
        self.assertEqual(predict_ucl(bundle, fixture, kickoff)[2], ["2026-11-23", "2026-11-22"])
        bundle["timeline"][2] = [(date(2026, 11, 22), 1500, history[:4])]
        self.assertIsNone(predict_ucl(bundle, fixture, kickoff, observed_at=early))


if __name__ == "__main__":
    unittest.main()
