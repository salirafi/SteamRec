-- tables created here are used by the fetch_item_similarity() function in recommender.py

-- IMPORTANT!
-- expected database names from config.py:
--   - database_similarity_table:  GAME_SIMILARITY and GAME_SCORES tables
--   - database_game_table: GAME_DATA table (must already exist)
-- current CSV inputs:
--   - tables/production/GAME_DATA.csv
--   - tables/production/GAME_SIMILARITY.csv
--   - tables/production/GAME_SCORES.csv

-- change file paths to your local file paths


--   mysql --local-infile=1 -u root -p < sql_script/load_tables.sql

SET GLOBAL LOCAL_INFILE = 1;

CREATE DATABASE IF NOT EXISTS steam_recommender;
USE steam_recommender;


-- ====================================================================================
-- GAME_SCORES
-- ====================================================================================

DROP TABLE IF EXISTS GAME_DATA;
CREATE TABLE IF NOT EXISTS GAME_DATA (
    item_id INT UNSIGNED PRIMARY KEY,
    item_name TEXT NOT NULL,
    release_date DATE,
    owners_count INT UNSIGNED,
    price FLOAT,
    median_playtime_forever FLOAT,
    user_reviews INT UNSIGNED,
    rating TINYINT(1) UNSIGNED,
    tags TEXT,
    developers TEXT,
    publishers TEXT,
    INDEX idx_item_id (item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

TRUNCATE TABLE GAME_DATA;

LOAD DATA LOCAL INFILE '/Users/salirafi/Documents/Personal Project/Steam Recommender/tables/production/GAME_DATA.csv'
INTO TABLE GAME_DATA
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(item_id, item_name, release_date, owners_count, price, median_playtime_forever, user_reviews, rating, tags, developers, publishers);


-- ====================================================================================
-- GAME_SIMILARITY
-- ====================================================================================

DROP TABLE IF EXISTS GAME_SIMILARITY;
CREATE TABLE IF NOT EXISTS GAME_SIMILARITY (
    source_item_id INT UNSIGNED NOT NULL,
    similar_item_id INT UNSIGNED NOT NULL,
    similarity_score FLOAT NOT NULL,
    PRIMARY KEY (source_item_id, similar_item_id),
    INDEX idx_source_item_id (source_item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


TRUNCATE TABLE GAME_SIMILARITY;


LOAD DATA LOCAL INFILE '/Users/salirafi/Documents/Personal Project/Steam Recommender/tables/production/GAME_SIMILARITY.csv'
INTO TABLE GAME_SIMILARITY
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(source_item_id, similar_item_id, similarity_score);



-- ====================================================================================
-- GAME_SCORES
-- ====================================================================================

DROP TABLE IF EXISTS GAME_SCORES;
CREATE TABLE IF NOT EXISTS GAME_SCORES(
    item_id INT UNSIGNED PRIMARY KEY,
    pop_score FLOAT NOT NULL,
    quality_score FLOAT NOT NULL,
    age_score FLOAT NOT NULL,
    INDEX idx_item_id (item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


TRUNCATE TABLE GAME_SCORES;


LOAD DATA LOCAL INFILE '/Users/salirafi/Documents/Personal Project/Steam Recommender/tables/production/GAME_SCORES.csv'
INTO TABLE GAME_SCORES
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(item_id, pop_score, quality_score, age_score);




-- ====================================================================================
-- CHECKS
-- ====================================================================================

-- -- verify item_similarity table
-- SELECT 'item_similarity' AS table_name, COUNT(*) AS row_count FROM item_similarity
-- UNION ALL
-- -- verify item_rerank_scores table
-- SELECT 'item_rerank_scores' AS table_name, COUNT(*) AS row_count FROM item_rerank_scores;

-- -- sample query from fetch_item_similarity() to verify data
-- SELECT
--     r.source_item_id,
--     r.similar_item_id AS item_id,
--     g.item_name,
--     r.similarity_score,
--     rk.pop_score,
--     rk.quality_score,
--     rk.age_score
-- FROM item_similarity r
-- JOIN GAME_DATA g ON g.item_id = r.similar_item_id
-- JOIN item_rerank_scores rk ON rk.item_id = r.similar_item_id
-- WHERE r.source_item_id = 10
-- ORDER BY r.similarity_score DESC
-- LIMIT 10;
