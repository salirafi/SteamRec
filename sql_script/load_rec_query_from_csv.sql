-- disable need primary key at table creation

CREATE DATABASE IF NOT EXISTS steam_rec_query;

USE steam_rec_query;


-- DROP TABLE IF EXISTS interaction_scores;
-- CREATE TABLE interaction_scores (
--     steam_id                        INT UNSIGNED NOT NULL,
--     item_id                         INT UNSIGNED NOT NULL,
--     interaction_score               FLOAT,
-- ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- DROP TABLE IF EXISTS als_user_factors;
-- CREATE TABLE als_user_factors (
--     steam_id                        INT UNSIGNED NOT NULL,
--     als_user_idx                    INT UNSIGNED NOT NULL,
--     factors                         JSON NOT NULL,
-- ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- DROP TABLE IF EXISTS als_item_factors;
-- CREATE TABLE als_item_factors (
--     item_id                         INT UNSIGNED NOT NULL,
--     als_item_idx                    INT UNSIGNED NOT NULL,
--     factors                         JSON NOT NULL
-- ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- -- DROP TABLE IF EXISTS als_user_index;
-- -- CREATE TABLE als_user_index (
-- --     steam_id                        INT UNSIGNED NOT NULL,
-- --     als_user_idx                    INT UNSIGNED NOT NULL,
-- -- ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- DROP TABLE IF EXISTS als_item_index;
-- CREATE TABLE als_item_index (
--     item_id                         INT UNSIGNED NOT NULL,
--     als_item_idx                    INT UNSIGNED NOT NULL
-- ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;


-- only scores from the cosine similarity, not re-ranked scores
DROP TABLE IF EXISTS item_similarity;
CREATE TABLE item_similarity (
    source_item_id              INT UNSIGNED NOT NULL,
    similar_item_id             INT UNSIGNED NOT NULL,
    similarity_rank             INT UNSIGNED NOT NULL,
    similarity_score            FLOAT NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- -- only re-ranked scores, not similarity scores
-- DROP TABLE IF EXISTS item_rerank_scores;
-- CREATE TABLE item_rerank_scores (
--     item_id INT                 UNSIGNED NOT NULL,
--     pop_score                   FLOAT,
--     quality_score               FLOAT,
--     age_score                   FLOAT
-- ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;


SET FOREIGN_KEY_CHECKS = 0;
SET UNIQUE_CHECKS = 0;


-- LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/interaction_scores.csv'
-- INTO TABLE interaction_scores
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 LINES;

-- LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/als_user_factors.csv'
-- INTO TABLE als_user_factors
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 LINES;

-- LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/als_item_factors.csv'
-- INTO TABLE als_item_factors
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 LINES;

-- LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/als_user_index.csv'
-- INTO TABLE als_user_index
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 LINES;

-- LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/als_item_index.csv'
-- INTO TABLE als_item_index
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 LINES;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/item_similarity.csv'
INTO TABLE item_similarity
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

-- LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/item_rerank_scores.csv'
-- INTO TABLE item_rerank_scores
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 LINES;

SET FOREIGN_KEY_CHECKS = 1;
SET UNIQUE_CHECKS = 1;


-- adding indexes

-- ALTER TABLE interaction_scores ADD PRIMARY KEY (steam_id, item_id),
--     ADD KEY idx_interaction_scores_item_id (item_id);

-- ALTER TABLE als_user_factors ADD PRIMARY KEY (steam_id),
--     ADD KEY idx_als_user_factors_idx (als_user_idx);

-- ALTER TABLE als_item_factors ADD PRIMARY KEY (item_id),
--     ADD KEY idx_als_item_factors_idx (als_item_idx);

-- ALTER TABLE als_user_index ADD PRIMARY KEY (steam_id),
--     ADD KEY idx_als_user_index_idx (als_user_idx);

-- ALTER TABLE als_item_index ADD PRIMARY KEY (item_id),
--     ADD KEY idx_als_item_index_idx (als_item_idx);

ALTER TABLE item_similarity ADD PRIMARY KEY (source_item_id, similar_item_id),
    ADD KEY idx_item_similarity_rank (source_item_id, similarity_rank),
    ADD KEY idx_item_similarity_target (similar_item_id);

-- ALTER TABLE item_rerank_scores ADD PRIMARY KEY (item_id);