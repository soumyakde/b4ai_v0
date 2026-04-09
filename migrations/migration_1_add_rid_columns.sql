-- =============================================================================
-- Migration 1: Add RID columns (no behavior change)
-- =============================================================================
-- Purpose:   Introduce a Research Identifier (RID) column to every table
--            that holds student research data. No existing columns are
--            modified or removed. The system continues to function
--            identically after this migration.
--
-- Applies to: responses.db (all research tables)
--             users.db     (users table only)
--
-- Safe to run multiple times — each ALTER TABLE is wrapped in a
-- conditional check in the companion Python script.
--
-- Run order:  Execute users.db block first, then responses.db block.
--             Do NOT run both in the same sqlite3 session.
--
-- Rollback:   SQLite cannot DROP COLUMN before version 3.35.
--             Rollback = restore from pre-migration backup.
--             Always run backup_databases() before executing.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- FILE: users.db
-- Connect to users.db before running this block.
-- -----------------------------------------------------------------------------

-- RID is nullable TEXT — populated by the backfill script after this migration.
-- UNIQUE constraint is deferred until Migration 3 (when all rows are populated).
ALTER TABLE users ADD COLUMN rid TEXT;


-- -----------------------------------------------------------------------------
-- FILE: responses.db
-- Connect to responses.db before running this block.
-- Run each ALTER TABLE separately — SQLite does not support multi-statement
-- ALTER TABLE in a single call.
-- -----------------------------------------------------------------------------

-- Core response and completion tables (direct student write paths)
ALTER TABLE responses         ADD COLUMN rid TEXT;
ALTER TABLE completions       ADD COLUMN rid TEXT;
ALTER TABLE assessment_scores ADD COLUMN rid TEXT;
ALTER TABLE survey_scores     ADD COLUMN rid TEXT;

-- Analytics output tables (populated by admin analysis runs)
ALTER TABLE cpi_summary       ADD COLUMN rid TEXT;
ALTER TABLE cpi_qual_scores   ADD COLUMN rid TEXT;
ALTER TABLE dta_results       ADD COLUMN rid TEXT;
ALTER TABLE dta_lo_results    ADD COLUMN rid TEXT;

-- Transcripts: participant_id becomes RID in Migration 3.
-- uploaded_by is admin identity — never receives a RID.
ALTER TABLE transcripts       ADD COLUMN rid TEXT;


-- =============================================================================
-- Verification queries (run after migration to confirm columns exist)
-- =============================================================================

-- In users.db:
-- SELECT name FROM pragma_table_info('users') WHERE name = 'rid';
-- Expected: 1 row

-- In responses.db:
-- SELECT name FROM pragma_table_info('responses')         WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('completions')       WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('assessment_scores') WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('survey_scores')     WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('cpi_summary')       WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('cpi_qual_scores')   WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('dta_results')       WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('dta_lo_results')    WHERE name = 'rid';
-- SELECT name FROM pragma_table_info('transcripts')       WHERE name = 'rid';
-- Expected: 9 rows (one per table)
