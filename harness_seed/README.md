# H-WAP harness store seed

Copied into a new versioned store by `VersionedStore.from_seed(root, "harness_seed")`.
The store is its own git repository; accepted self-edits are commits with evidence,
and `changelog.jsonl` also records rejected/pending changes and rollbacks.

- `memory/<kind>/<id>.md`: Markdown notes (kinds: episode, world, body, world_model, perception).
- `skills/<name>/`: `SKILL.md`, `examples/`, optional `scripts/` helpers (JSON stdin -> keyframes stdout).
- `guidance/{planner,critic,perception}.md`: harness guidance edited by the outer loop.
