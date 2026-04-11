# SteamRec: A Recommender for Steam

This repository contains the source code of a recommender system for the popular game store [Steam](https://store.steampowered.com/about/) using a combination of collaborative and content-based filtering technique, made with Python. It covers an end-to-end pipeline from data collection to web app building. 


🎥 [YOU CAN ACCESS THE LIVE DEMO HERE](https://steam-rec.vercel.app/) 🎥


![/figures/image.png](/figures/image.png)


## Introduction

For excellent exercise on building a recommender system from scratch with Python, the reader can go to this <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"><img alt="GitHub Invertocat" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20"></picture> [Github repo](https://github.com/topspinj/recommender-tutorial). This includes making a recommender system with collaborative filtering technique using [k Nearest Neighbor (kNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) and [Alternating Least Square (ALS)](https://github.com/benfred/implicit), as well as the content-based filtering using cosine similarity.

At its heart, a recommender system relies on two types of data: the items and user interactions (for collaborative-filtering technique). In this project, items are games available on Steam whilst user interactions could take various forms such as whether the user recommends the game, their total playtime, what's in their wishlist, etc. For content-based filtering technique, the recommender will filter, out of the whole catalog, games that are most similar to the game in question based on their contents such as tags and genres. For collaborative-filtering technique, the recommender will evaluate whether a user would like a game based on other similar users that play similar games. Following this, the deployed app uses two complementary recommendation paths:
- A content-based recommender that retrieves similar games based on games' metadata.
- A collaborative-filtering recommender that folds a live Steam library into a pretrained ALS latent space and ranks the games from the item latent factors.
The frontend layer then has two search modes based on these two techniques:
- `Game-based search`: enter a game title, retrieve the closest catalog neighbors, and rerank them.
- `Steam-ID search`: fetch a user's owned games from the Steam Web API, infer interaction strength from playtime, fold the user into the ALS space, and rerank the candidate pool.

The datasets that are used to build the system are:
- [Steam Games Dataset from Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) (per 2025) for the game data details,
- [100 Million+ Steam Reviews from Kaggle](https://www.kaggle.com/datasets/e2355a9b846ac37e77dc85210d20656dc8c20f349b7c30d6b6433348e959c484) (per 2025) for the user reviews.


## Stack

### Backend

- Python
- Flask
- Pandas
- NumPy
- SciPy
- scikit-learn
- implicit
- MySQL

### Frontend

- HTML
- React JS
- CSS


## Running

All commands below assume the terminal is opened at the repository root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare raw inputs

Place the raw source files in [`tables/raw`](tables/raw) (see [Data Inputs](#data-inputs)):

- `games.json`
- `all_reviews.csv`

### 3. Generate production CSV files

```bash
python src/process_game_data.py
python src/process_game_review.py
python src/recommender_matrices.py
```

This will run for a while since the reviews table is huge and ALS training takes some time. In my Macbook M1 Max, it takes ~45 mins.

### 4. Load serving tables into MySQL

```bash
mysql --local-infile=1 -u root -p < sql_script/load_tables.sql
```

Before running, change the file path to `your/local/path/to/tables/production/`.

### 5. Start the app

```bash
python app.py
```

The development server runs on:

- `http://127.0.0.1:8000`


## Configuration

Make a `.env` file with the following variables for MySQL credentials and database names.

### Required database variables

| Variable | Description | Example |
| --- | --- | --- |
| `STEAM_DB_HOST` | MySQL host | `localhost` |
| `STEAM_DB_PORT` | MySQL port | `3306` |
| `STEAM_DB_USER` | MySQL username | `root` |
| `STEAM_DB_PASSWORD` | MySQL password | `your-password` |
| `STEAM_DB_NAME` | Database name used by the app | `steam_recommender` |

### Required Steam API variable

| Variable | Description | Example |
| --- | --- | --- |
| `STEAM_WEB_API_KEY` | Steam Web API key for owned-games lookup | `your-steam-api-key` |

### Recommender weights and model settings

Default recommendation settings live in [`src/config.py`](src/config.py):

- `w_cb = 1.0`
- `w_cf = 0.7`
- `w_age = 0.7`
- `w_popularity = 1.3`
- `w_quality = 1.0`
- `als_factors = 64`
- `als_iterations = 20`
- `als_regularization = 0.1`
- `als_alpha = 1.0`


## Structure

```text
Steam Recommender/
├── app.py
├── README.md
├── requirements.txt
├── sql_script/
│   └── load_tables.sql
├── src/
│   ├── _get_steam_API.py
│   ├── config.py
│   ├── helpers.py
│   ├── process_game_data.py
│   ├── process_game_review.py
│   ├── recommender.py
│   └── recommender_matrices.py
├── static/
│   ├── css/steam_recommender.css
│   ├── github.svg
│   └── js/steam_recommender.js
├── templates/
│   └── steam_recommender.html
├── tables/
│   ├── raw/
│   └── production/
└── notebooks/
    └── recommendation_normalized_hours_eda.ipynb
```

## Data Inputs

The pipeline expects raw source files under [`tables/raw`](tables/raw) and these can be downloaded from [HERE][Steam Games Dataset from Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) for the game details and [HERE](https://www.kaggle.com/datasets/e2355a9b846ac37e77dc85210d20656dc8c20f349b7c30d6b6433348e959c484) for the user reviews. Alternatively using Kaggle CLI
``` 
#!/bin/bash
kaggle datasets download fronkongames/steam-games-dataset
kaggle datasets download kieranpoc/steam-reviews
```

The code reads `games.json` and `all_reviews.csv` directly under the [`tables/raw`](tables/raw) folder. 


## Recommendation Logic

### Content-based Filtering

The content-based recommender works only from item metadata and does not require any user history. In the current implementation, the similarity model is built primarily from each game's Steam tags. Each game is represented as a sparse tag vector, the vectors are L2-normalized, and item-to-item similarity is computed with the kNN method.

The content-based pipeline is then:

1. Build a sparse tag matrix for all games and normalize each row.
2. For each game, determine the $k$ closest neighbors against the rest of the catalog. This stores roughly $n_{\text{games}} \times k$ item-item similarity rows.
3. Compute three additional item-level scores used for re-ranking:
   - Popularity score, derived from the log-transformed number of user reviews.
   - Quality score, derived from the positive review ratio.
   - Age score, derived from the inverse of the game's age, so newer games receive a higher score.
4. Re-rank the top $k$ similar items using a weighted sum. In the implementation, the base similarity score is normalized before being combined with the other re-rank features:

```math
\text{final\_score}_i
=
w_{\text{sim}} \,\hat{s}_i
+ w_{\text{pop}} \, p_i
+ w_{\text{qual}} \, q_i
+ w_{\text{age}} \, a_i
```

where $\hat{s}_i$ is the normalized similarity score for candidate item $i$, $p_i$ is its popularity score, $q_i$ is its quality score, and $a_i$ is its age score.

> In short, the recommender first finds items that are content-wise similar, then the final ranking layer makes the output more practical by balancing personalization with popularity, review quality, and recency.


### Collaborative filtering


The collaborative-filtering recommender learns from user-item interactions rather than game metadata. The implementation uses [ALS](https://github.com/benfred/implicit) for implicit-feedback recommendation, following the framework introduced by [Hu, Koren, and Volinsky (2008)](http://yifanhu.net/PUB/cf.pdf).

Instead of explicit ratings, the model uses interaction strength derived from playtime and the binary recommend flag in the review dataset. For each observed user-game pair $(u, i)$, the pipeline builds an interaction score:

```math
r_{ui} = \log(1 + h_{ui}) \, m_{ui} \,e/{ui}
```

where $h_{ui}$ is the user's normalized by median game playtime (total playtime divided by the game's median total playtime in hours),

```math
m_{ui} =
\begin{cases}
1.5 & \text{if the user recommends the game} \\
0.5 & \text{if the user does not recommend the game} \\
1.0 & \text{if no data}
\end{cases}
```
and,
```math
e_{ui} =
\begin{cases}
0.8 & \text{if the game is in early access} \\
1.0 & \text{if the game is not in early access}
\end{cases}
```

The ALS confidence value is then constructed as:

```math
c_{ui} = 1 + \alpha r_{ui}
```

This produces a sparse confidence matrix whose shape is $(n_{\text{users}}, n_{\text{items}})$.

The collaborative-filtering pipeline is then:

1. Build the interaction score $r_{ui}$ from playtime and the recommendation flag.
2. Convert it into an implicit-feedback confidence matrix using $c_{ui} = 1 + \alpha r_{ui}$.
3. Train ALS to learn latent factors for users and items. In practice, the pipeline stores only the item latent factors and reconstructs a live user's latent vector at request time.
4. When a user enters a Steam ID from the web app, it will fetch their owned games from [Steam Web API: `IPlayerService/GetOwnedGames`](https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/). 
    > ⚠️ IMPORTANT ⚠️ Since the endpoint provides ownership and playtime but not review recommendations, the current online pipeline treats owned games as positive interactions when forming live inputs. This behavior is actually not proper, used only for convenience. In the future, this will be patched.
5. Fold the live user into the pretrained item-factor space. If $Y$ is the matrix of item latent factors, the user vector $\mathbf{x}_u$ is solved from the observed items by a regularized linear system of the form:

```math
\mathbf{x}_u
=
\left(
Y^\top Y
+ \sum_{i \in \mathcal{I}_u} (c_{ui} - 1)\,\mathbf{y}_i \mathbf{y}_i^\top
+ \lambda I
\right)^{-1}
\left(
\sum_{i \in \mathcal{I}_u} c_{ui}\,\mathbf{y}_i
\right)
```

where $\mathcal{I}_u$ is the set of items observed for user $u$, $\mathbf{y}_i$ is the latent factor vector of item $i$, and $\lambda$ is the regularization coefficient. This is the fold-in step used by the live recommender.
6. Score each candidate item by dot product with the folded-in user vector:

```math
\text{cf\_score}_i = \mathbf{y}_i^\top \mathbf{x}_u
```

7. Exclude items already seen by the user, keep a candidate pool of the highest-scoring items, and then apply the same re-ranking framework used by the content-based recommender:

```math
\text{final\_score}_i
=
w_{\text{cf}} \,\widehat{\text{cf\_score}}_i
+ w_{\text{pop}} \, p_i
+ w_{\text{qual}} \, q_i
+ w_{\text{age}} \, a_i
```

where the collaborative-filtering score is normalized before it is combined with the popularity, quality, and age parameters.

> In short, the collaborative model first captures user taste in a latent space, then the final ranking layer makes the output more practical by balancing personalization with popularity, review quality, and recency.



## Steam API Integration

[`src/_get_steam_API.py`](src/_get_steam_API.py) calls:

- `https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/`




## Sources

Reading sources:

- [Hu, Koren, and Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*](http://yifanhu.net/PUB/cf.pdf)
- [Koren, Bell, and Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems*](https://ieeexplore.ieee.org/document/5197422)
- [Rendle et al. (2009), *BPR: Bayesian Personalized Ranking from Implicit Feedback*](https://arxiv.org/abs/1205.2618)

Pipeline-specific implementation:

- [Steam Web API: `IPlayerService/GetOwnedGames`](https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/)
- [Game Recommendations on Steam Dataset (Kaggle)](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- [benfred/implicit](https://github.com/benfred/implicit)

## Author's Remarks

This is a personal project intended to be a portfolio. I am not currently planning to push into production except if there are some interested collaborators, in which case, please feel free to contact me at salirafi8@gmail.com :)

The use of generative AI includes: Github Copilot to help in code syntax and identifying bugs and errors. Outside of those, including problem formulation and framework of thinking, code logical reasoning and writing, from database management to web development, all is done mostly by the author.
