# SteamRec: A Recommender for Steam

This repository contains the source code of a recommender system for the popular game store [Steam](https://store.steampowered.com/about/) using a combination of collaborative and content-based filtering technique, made with Python. It covers an end-to-end pipeline from data collection to web app building. 

![/figures/home_page.png](/figures/home_page.png)


## Introduction

For excellent exercise on building a recommender system from scratch with Python, the reader can go to this <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"><img alt="GitHub Invertocat" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20"></picture> [Github repo](https://github.com/topspinj/recommender-tutorial). This includes making a recommender system with collaborative filtering technique using [k Nearest Neighbor (kNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) and [Alternating Least Square (ALS)](https://github.com/benfred/implicit), as well as the content-based filtering using cosine similarity.

At its heart, a recommender system relies on two types of data: the items and user interactions (for collaborative-filtering technique). In this project, items are games available on Steam whilst user interactions could take various forms such as whether the user recommends the game, their total playtime, what's in their wishlist, etc. For content-based filtering technique, the recommender will filter, out of the whole catalog, games that are most similar to the game in question based on their contents such as tags and genres. For collaborative-filtering technique, the recommender will evaluate whether a user would like a game based on other similar users that play similar games.

The recommender system is based originally on [Game Recommendations on Steam Dataset from Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (per 2022) 
> This dataset covers almost all games until year 2022 and contains more than 41 million user reviews and 50,000 games.

The database will be updated to up to 2026 in the future.

## Tools Used

### **Backend**

- Numpy
- Pandas
- Sklearn
- implicit
- Flask
- MySQL

### **Frontend**

- HTML
- React JavaScript

## Running

If the reader intents to run the whole pipeline (from data download to web app interaction), please follow the below steps. All steps assume the terminal is run from the project's root folder and the user has installed MySQL Server (as of this writing, MySQL version used is 8.0). 

Download the data from [HERE](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) or alternatively using Kaggle CLI
``` 
#!/bin/bash
kaggle datasets download antonkozyriev/game-recommendations-on-steam
```
and save them to [tables\raw\](tables\raw\).

The whole run should take no more than 30 mins, with more than half of it is attributed to the ALS training and user-item interactions table writing and reading.

### Windows

On Windows, the entire pipeline can be run from PowerShell
```
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_pipeline.ps1
.\scripts\run_app.ps1
```
Make sure to run it as administrator and to have all environment variables shown in [Environment Variables](#environment-variables) set.

### Others

**Note that the pipeline has not yet been tested in OS other than Windows** and there is no .sh script yet for running it conveniently. However, the reader can try to run this chain of commands although I don't guarantee there won't be an error.

```
python ./src/prepare_production_tables.py

rsync -a ".\tables\production" "\Path\to\MySQL\MySQL Server 8.0\Uploads" -v -u

mysql --user root --password --host localhost --port 3306 -e "source sql_script\load_production_tables.sql"

python ./src/prepare_recommender_matrices.py

rsync -a ".\tables\rec_matrices" "\Path\to\MySQL\MySQL Server 8.0\Uploads" -v -u

mysql --user root --password --host localhost --port 3306 -e "source sql_script\load_rec_query_from_csv.sql"

python app.py
```

## Environment Variables

Make a `.env` file with the following variables for MySQL credentials and database names.

| Variable | Example value | Description |
| --- | --- | --- |
| `STEAM_DB_USER` | `root` | MySQL username. |
| `STEAM_DB_PASSWORD` | `user-password` | MySQL password. |
| `STEAM_DB_HOST` | `localhost` | MySQL host address. |
| `STEAM_DB_PORT` | `3306` | MySQL port. |
| `STEAM_DB_QUERY_NAME` | `steam_rec_query` | Database used for recommender query artifacts. |
| `STEAM_DB_PROD_NAME` | `steam_rec` | Main production database containing the cleaned Steam data. |
| `MYSQL_UPLOAD_DIR` | `"C:/ProgramData/MySQL/MySQL Server 8.0/Uploads"` | Path to local MySQL server "Uploads" folder. |
| `API_KEY` | `API_KEY` | API key to Steam Web API. See [here](https://steamcommunity.com/dev). |

To look at the default path to "Uploads" folder for `MYSQL_UPLOAD_DIR`, run in MySQL
```
SHOW VARIABLES LIKE 'secure_file_priv';
```

Please note that the current SQL script to create the database is still hardcoding the database name to `steam_rec` in [\sql_script\load_production_tables.sql](\sql_script\load_production_tables.sql) and `steam_rec_query` in [\sql_script\load_rec_query_from_csv.sql](\sql_script\load_rec_query_from_csv.sql).

## Content


## Project Structure

```text
Steam-Recommendation-System/
├── app.py                         # Flask app entry point
├── README.md                      
├── LICENSE                        
│
├── src/                          # Core Python source code
│   ├── config.py
│   ├── helpers.py
│   ├── prepare_production_tables.py
│   ├── prepare_recommender_matrices.py
│   ├── recommender.py
│   ├── _get_steam_API.py
│   ├── __init__.py
│
├── templates/                      # HTML templates
│   └── steam_recommender.html
│
├── static/                         # Frontend static assets
│   ├── css/
│   ├── js/
│   └── github.svg
│
├── tables/                         # Data tables and generated datasets
│   ├── production/                 # will be created within the pipeline
│   ├── raw/
│   ├── rec_matrices/               # will be created within the pipeline
│
├── sql_script/                     # SQL scripts for production loading
│   ├── load_production_tables.sql
│   ├── load_rec_query_from_csv.sql
│
└── figures/                        # Evaluation charts and diagrams
    ├── EER_DIAGRAM_KAGGLE_ONLY.jpg
    └── ...
```

- `app.py`: Main application entry point.
- `src/`: Core backend logic and recommendation pipeline.
- `templates/`: HTML templates for the web interface.
- `static/`: Frontend assets such as CSS, JavaScript, and images.
- `tables/`: Stored raw, production, and recommendation data tables.
- `sql_script/`: SQL scripts for schema creation, checks, and data loading.
- `figures/`: Evaluation plots, charts, and project diagrams.

## Pipeline Description

The workflow is the following:
1. Data pre-processing (cleaning, parsing, exploding nested dicts, etc.) and exporting the resulting tables to CSV with `prepare_production_tables.py`.
2. Database loading with MySQL:
    - The pre-processed data is loaded to MySQL as the production tables which will be used primarily for calculations of recommender-essential matrices (see later).
    - The EER diagram of the production tables is shown below.
    ![EER Diagram](/figures/EER_DIAGRAM_KAGGLE_ONLY.png)
3. Building the item similarity matrix using cosine similarity for content-based filtering (`item_matrix` in `prepare_recommender_matrices.py`) and the user-item interaction matrix using ALS for the collaborative-filtering (expressed as the latent factors `user_factors` and `item_factors`) through database querying. The recommendations will be computed by recalling these matrices at the requested `steam_id` or `item_id` and then applying some weights that connected to the game's popularity, quality, similarity to the user's taste, and age to rank the recommendations. 
5. Back-end &#U+2194 front-end interaction for displaying the resulting recommendations to the user (that can choose between user-based and content-based recommender). These recommendations can be "re-ranked" by adjusting the four weights mentioned in 3.

There is also an evaluation step for the recommender's performance based on the Recall@K, NDCG@K, and Hit@K metrics. See branch `dev` for this.

## Building the Recommendation System

### Content-based Filtering

The content-based recommender works only from item metadata and does not require any user history. In the current implementation, the similarity model is built primarily from each game's Steam tags. Each game is represented as a sparse tag vector, the vectors are L2-normalized, and item-to-item similarity is computed with [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity):

```math
\cos(\theta) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
```

Here, $\mathbf{a}$ and $\mathbf{b}$ are the feature vectors of two games. Since the vectors are normalized before similarity is computed, the cosine score measures how similar the two games are in terms of their tag profiles.

The content-based pipeline is then:

1. Build a sparse tag matrix for all games and normalize each row.
2. For each game, compute cosine similarity against the rest of the catalog and keep only the top $k$ most similar items. This stores roughly $n_{\text{games}} \times k$ item-item similarity rows rather than the full dense matrix.
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

### Collaborative Filtering

The collaborative-filtering recommender learns from user-item interactions rather than game metadata. The implementation uses [ALS](https://github.com/benfred/implicit) for implicit-feedback recommendation, following the framework introduced by [Hu, Koren, and Volinsky (2008)](http://yifanhu.net/PUB/cf.pdf).

Instead of explicit ratings, the model uses interaction strength derived from playtime and the binary recommend flag in the review dataset. For each observed user-game pair $(u, i)$, the pipeline builds an interaction score:

```math
r_{ui} = \log(1 + h_{ui}) \, m_{ui}
```

where $h_{ui}$ is the user's playtime in hours and

```math
m_{ui} =
\begin{cases}
1.5 & \text{if the user recommends the game} \\
0.5 & \text{otherwise}
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

The use of generative AI includes: Github Copilot to help in code syntax and comments/docstring writing, as well as OpenAI's Chat GPT to help with identifying bugs and errors, and to also write the `.ps1` scripts per my guidance. Outside of those, including problem formulation and framework of thinking, code logical reasoning and writing, from database management to web development, all is done mostly by the author.


