-- AlignAI SQLite schema and sample queries for alignment dataset management.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    domain TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    source TEXT NOT NULL,
    token_estimate INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_variants (
    model_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL CHECK (strategy IN ('base', 'full', 'lora', 'qlora')),
    base_model TEXT NOT NULL,
    checkpoint_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    evaluation_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_variants(model_id),
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    judge_model TEXT NOT NULL,
    avg_judge_score REAL NOT NULL,
    alignment_readiness REAL NOT NULL,
    confidence_score REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS judge_scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(evaluation_id),
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    category TEXT NOT NULL,
    score REAL NOT NULL CHECK (score >= 1 AND score <= 5),
    justification TEXT
);

CREATE TABLE IF NOT EXISTS human_preferences (
    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    model_a TEXT NOT NULL REFERENCES model_variants(model_id),
    model_b TEXT NOT NULL REFERENCES model_variants(model_id),
    winner TEXT NOT NULL CHECK (winner IN ('model_a', 'model_b', 'tie', 'skip')),
    reviewer_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Sample query: dataset quality rollup by split.
-- SELECT split, COUNT(*) AS conversations, SUM(token_estimate) AS estimated_tokens
-- FROM conversations
-- GROUP BY split;

-- Sample query: average judge score by model strategy.
-- SELECT mv.strategy, AVG(js.score) AS avg_category_score
-- FROM judge_scores js
-- JOIN evaluation_runs er ON er.evaluation_id = js.evaluation_id
-- JOIN model_variants mv ON mv.model_id = er.model_id
-- GROUP BY mv.strategy
-- ORDER BY avg_category_score DESC;

-- Sample query: human preference win rate by model.
-- SELECT model_id, AVG(win) AS win_rate
-- FROM (
--   SELECT model_a AS model_id, CASE WHEN winner = 'model_a' THEN 1.0 WHEN winner = 'tie' THEN 0.5 ELSE 0.0 END AS win
--   FROM human_preferences
--   UNION ALL
--   SELECT model_b AS model_id, CASE WHEN winner = 'model_b' THEN 1.0 WHEN winner = 'tie' THEN 0.5 ELSE 0.0 END AS win
--   FROM human_preferences
-- )
-- GROUP BY model_id
-- ORDER BY win_rate DESC;
