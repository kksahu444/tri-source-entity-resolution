# Section 8 leak check (26 Sep, dev universe, city-grouped folds)

Leak share = share of validation candidate pairs whose record's true S1 is a dev query in another fold.

| Grouping | India | US | Pairs measured |
|---|---|---|---|
| (country, city_key), 5 folds (in use) | 0.0699 | 0.0858 | all union pairs (features stage log, runs/features_dev.json) |
| (country, city_key), 5 folds | 0.0615 | 0.0582 | union rank <= 10 |
| (country, region), 5 folds, 78 groups | 0.0556 | 0.0611 | union rank <= 10 |

Coarsening the groups from cities to states does not bring the share under 5%, so the cross-group
candidates are not a city-parsing artefact: they are name-driven candidates (same or similar business
name in another state, often records whose address is truncated or empty). Removing them would need
name-based grouping, which chains into one giant group.

Why this does not leak labels: every feature is a similarity or rank computed from the two records'
text and the blocking scores; no feature carries an entity or record identity, so a model cannot
learn "record X belongs to S1 B" in a training fold and reuse it when X appears as a negative for S1 A.

Status: flagged for the human (spec: fix grouping if > 5%). City grouping kept.
