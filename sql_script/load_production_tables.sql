-- disable need primary key at table creation

CREATE DATABASE IF NOT EXISTS steam_rec;

USE steam_rec;

-- =====================================================
-- CREATE THE PRODUCTION TABLES
-- =====================================================


DROP TABLE IF EXISTS GAME_DATA;
CREATE TABLE GAME_DATA (
    item_id        INT UNSIGNED NOT NULL,
    item_name      VARCHAR(255),
    release_date   DATE,
    price_original FLOAT,
    price_final    FLOAT,
    discount       FLOAT,
    rating         TINYINT UNSIGNED, -- 1-9 per SENTIMENT_MAP
    positive_ratio TINYINT UNSIGNED, -- 0-100
    user_reviews   MEDIUMINT UNSIGNED,
    PRIMARY KEY (item_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;



DROP TABLE IF EXISTS GAME_TAG;
CREATE TABLE GAME_TAG (
    item_id INT UNSIGNED NOT NULL,
    tag     VARCHAR(100) NOT NULL,
    PRIMARY KEY (item_id, tag)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS GAME_USER;
CREATE TABLE GAME_USER (
    steam_id INT UNSIGNED NOT NULL, -- this steam_id is not real steam_id, it's just an auto incremented integer for the Kaggle dataset
    products MEDIUMINT UNSIGNED,
    reviews  MEDIUMINT UNSIGNED,
    PRIMARY KEY (steam_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;


-- keys are assigned after loading to speed it up
DROP TABLE IF EXISTS GAME_REVIEW;
CREATE TABLE GAME_REVIEW (
    review_id			INT UNSIGNED NOT NULL,
    item_id				INT UNSIGNED NOT NULL,
    steam_id			INT UNSIGNED NOT NULL,
    hours				FLOAT,
    date				DATE,
    recommend			BOOLEAN,
    helpful				MEDIUMINT UNSIGNED
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;



-- =====================================================
-- POPULATE THE PRODUCTION TABLES
-- `LOAD DATA INFILE` must live under @@secure_file_priv if that variable is set:
--     SHOW VARIABLES LIKE 'secure_file_priv';
-- =====================================================


SET NAMES utf8mb4;

-- disabling checks for faster builk loading; will re-enable after loading
SET UNIQUE_CHECKS = 0;
SET FOREIGN_KEY_CHECKS = 0;

-- delete existing rows
TRUNCATE TABLE GAME_REVIEW;
TRUNCATE TABLE GAME_USER;
TRUNCATE TABLE GAME_DATA;
TRUNCATE TABLE GAME_TAG;


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/GAME_TAG.csv'
INTO TABLE GAME_TAG
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/GAME_DATA.csv'
INTO TABLE GAME_DATA
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/GAME_USER.csv'
INTO TABLE GAME_USER
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/GAME_REVIEW.csv'
INTO TABLE GAME_REVIEW
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

SET UNIQUE_CHECKS = 1;
SET FOREIGN_KEY_CHECKS = 1;

ALTER TABLE GAME_REVIEW ADD PRIMARY KEY (review_id);
ALTER TABLE GAME_REVIEW ADD UNIQUE KEY (steam_id, item_id); -- there should be no duplicates since they have been handled in prepare_production_tables.py
