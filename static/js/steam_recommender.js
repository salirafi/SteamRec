const { useMemo, useState } = React;

const INITIAL_WEIGHTS = {
  popularity: 0.5,
  quality: 0.7,
  age: 0.4,
  similarity: 0.65,
};

const PROJECT_REPO_URL = "https://github.com/salirafi/SteamRec";
const STEAM_ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/8/83/Steam_icon_logo.svg";

const RATING_MAP = {
  "Overwhelmingly Positive": "positive",
  "Very Positive": "positive",
  "Mostly Positive": "positive",
  "Positive": "positive",
  "Mixed": "mixed",
  "Mostly Negative": "negative",
  "Negative": "negative",
  "Very Negative": "negative",
  "Overwhelmingly Negative": "negative",
};

function formatNum(n) {
  return new Intl.NumberFormat("en-US").format(n || 0);
}

function formatPeople(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return "Unknown";
  }
  return values.join(", ");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();

  let data = {};
  if (contentType.includes("application/json")) {
    data = rawText ? JSON.parse(rawText) : {};
  } else if (rawText) {
    throw new Error(rawText.slice(0, 200));
  }

  if (!response.ok) {
    throw new Error(data.message || `Request failed with status ${response.status}`);
  }

  return data;
}

function GameSuggestion({ game, onSelect }) {
  return (
    <button className="suggestion-item" type="button" onClick={() => onSelect(game)}>
      <img className="suggestion-image" src={game.image} alt={game.name} loading="lazy" />
      <span className="suggestion-name">{game.name}</span>
    </button>
  );
}

function TopBar({ onHomeClick, showHomeButton = false }) {
  return (
    <nav className="nav">
      <div className="nav-logo">SteamRec <span>Recommender</span></div>
      <div className="nav-right">
        {showHomeButton && (
          <>
            <button className="back-btn" onClick={onHomeClick}>New Search</button>
          </>
        )}
        <div className="divider" />
        <a className="repo-link" href={PROJECT_REPO_URL} target="_blank" rel="noopener noreferrer">
          <img className="repo-icon" src="/static/github.svg" alt="" aria-hidden="true" />
          Go to project repo
        </a>
        <div className="divider" />
        <div className="nav-badge">steamrec@v2.0</div>
      </div>
    </nav>
  );
}

function GameCard({ game }) {
  const tags = (game.tags || []).slice(0, 6);

  return (
    <div className="game-card">
      <span className="rank-badge">#{game.rank}</span>
      {game.image ? (
        <img className="card-img" src={game.image} alt={game.name} loading="lazy" />
      ) : (
        <div className="card-img-placeholder">?</div>
      )}

      <div className="card-body">
        <div className="card-name">{game.name}</div>
        <div className="card-id">APP ID: {game.id}</div>

        <div className="card-meta">
          <div className="meta-stack">
            <span className={`meta-pill ${RATING_MAP[game.rating] || "mixed"}`}>{game.rating}</span>
            <span className="meta-reviews">{formatNum(game.userReviews)} user reviews</span>
          </div>

          <div className="score-chip">
            <span className="score-chip-label">RECOMMENDATION SCORE</span>
            <span className="score-chip-value">{Number(game.recommendationScore || 0).toFixed(3)}</span>
            <a className="steam-link" href={game.steamUrl} target="_blank" rel="noopener noreferrer">
              <img className="steam-link-icon" src={STEAM_ICON_URL} alt="" aria-hidden="true" />
              Go to Steam
            </a>
          </div>
        </div>

        <div className="card-studio">
          <div className="studio-line">
            <span className="studio-label">Released</span>
            <span className="studio-value">{game.releaseDate || "Unknown"}</span>
          </div>
          <div className="studio-line">
            <span className="studio-label">Developers</span>
            <span className="studio-value">{formatPeople(game.developers)}</span>
          </div>
          <div className="studio-line">
            <span className="studio-label">Publishers</span>
            <span className="studio-value">{formatPeople(game.publishers)}</span>
          </div>
        </div>

        {tags.length > 0 && (
          <div className="card-tags">
            {tags.map((tag) => (
              <span key={`${game.id}-${tag}`} className="card-tag">{tag}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SliderRow({ label, desc, value, onChange, onRelease }) {
  const pct = `${value * 100}%`;

  return (
    <div className="slider-item">
      <div className="slider-header">
        <span className="slider-name">{label}</span>
        <span className="slider-value">{value.toFixed(2)}</span>
      </div>
      <div className="slider-desc">{desc}</div>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={value}
        style={{ "--pct": pct }}
        onChange={(event) => onChange(Number(event.target.value))}
        onMouseUp={onRelease}
        onTouchEnd={onRelease}
        onKeyUp={onRelease}
      />
    </div>
  );
}

function App() {
  const [page, setPage] = useState("home");
  const [mode, setMode] = useState("game");
  const [inputValue, setInputValue] = useState("");
  const [searchContext, setSearchContext] = useState(null);
  const [games, setGames] = useState([]);
  const [weights, setWeights] = useState(INITIAL_WEIGHTS);
  const [activeTags, setActiveTags] = useState(new Set());
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [gameSuggestions, setGameSuggestions] = useState([]);
  const [gameSearchLoading, setGameSearchLoading] = useState(false);
  const [selectedGame, setSelectedGame] = useState(null);

  const allTagCounts = useMemo(() => {
    const counts = {};
    games.forEach((game) => {
      (game.tags || []).forEach((tag) => {
        counts[tag] = (counts[tag] || 0) + 1;
      });
    });
    return counts;
  }, [games]);

  const sortedTags = useMemo(
    () => Object.entries(allTagCounts).sort((a, b) => b[1] - a[1]).map(([tag]) => tag),
    [allTagCounts]
  );

  const visibleGames = useMemo(() => {
    if (activeTags.size === 0) {
      return games;
    }
    return games.filter((game) =>
      (game.tags || []).some((tag) => activeTags.has(tag))
    );
  }, [activeTags, games]);

  function toggleTag(tag) {
    setActiveTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) {
        next.delete(tag);
      } else {
        next.add(tag);
      }
      return next;
    });
  }

  function resetTags() {
    setActiveTags(new Set());
  }

  async function runUserSearch(steamId = inputValue, currentWeights = weights) {
    if (!steamId) return;

    setLoading(true);
    setError("");
    setMessage("");
    setPage("loading");

    try {
      const data = await fetchJson("/api/recommend/user", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steam_id: steamId,
          weights: {
            popularity: currentWeights.popularity,
            quality: currentWeights.quality,
            age: currentWeights.age,
            similarity: currentWeights.similarity,
          },
        }),
      });

      setSearchContext({
        mode: "user",
        id: steamId,
        label: `Steam User ${steamId}`,
        searchId: data.search_id || null,
      });
      setGames(data.results || []);
      setActiveTags(new Set());
      setMessage(data.message || "");
      setPage("results");
    } catch (err) {
      setGames([]);
      setError(err.message || "Failed to fetch recommendations.");
      setPage("results");
    } finally {
      setLoading(false);
    }
  }

  async function runGameSearch(game, currentWeights = weights) {
    if (!game?.id) return;

    setLoading(true);
    setError("");
    setMessage("");
    setPage("loading");

    try {
      const data = await fetchJson("/api/recommend/game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          item_id: Number(game.id),
          weights: {
            popularity: currentWeights.popularity,
            quality: currentWeights.quality,
            age: currentWeights.age,
            similarity: currentWeights.similarity,
          },
        }),
      });

      setSelectedGame(game);
      setInputValue(game.name);
      setGameSuggestions([]);
      setSearchContext({
        mode: "game",
        id: String(game.id),
        label: game.name,
        searchId: data.search_id || null,
      });
      setGames(data.results || []);
      setActiveTags(new Set());
      setMessage(data.message || "");
      setPage("results");
    } catch (err) {
      setGames([]);
      setError(err.message || "Failed to fetch recommendations.");
      setPage("results");
    } finally {
      setLoading(false);
    }
  }

  async function fetchGameSuggestions(query) {
    if (mode !== "game") {
      return;
    }

    const trimmed = query.trim();
    if (!trimmed) {
      setGameSuggestions([]);
      setGameSearchLoading(false);
      return;
    }

    setGameSearchLoading(true);
    setError("");
    setMessage("");

    try {
      const data = await fetchJson(`/api/search/games?q=${encodeURIComponent(trimmed)}`);
      setGameSuggestions(data.results || []);
      if (!data.results || data.results.length === 0) {
        setError("No matching games found. Try a different title.");
      }
    } catch (err) {
      setGameSuggestions([]);
      setError(err.message || "Failed to fetch game options.");
    } finally {
      setGameSearchLoading(false);
    }
  }

  function handleSearch() {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    if (mode === "user") {
      runUserSearch(trimmed, weights);
      return;
    }

    if (selectedGame && selectedGame.name === trimmed) {
      runGameSearch(selectedGame, weights);
      return;
    }

    fetchGameSuggestions(trimmed);
  }

  function handleKey(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSearch();
    }
  }

  function handleInputChange(value) {
    setInputValue(value);
    setError("");
    setMessage("");
    setGameSuggestions([]);
    setGameSearchLoading(false);
    if (!selectedGame || selectedGame.name !== value) {
      setSelectedGame(null);
    }
  }

  function handleModeChange(nextMode) {
    setMode(nextMode);
    setInputValue("");
    setSelectedGame(null);
    setGameSuggestions([]);
    setError("");
    setMessage("");
  }

  function setWeight(key) {
    return (value) => {
      setWeights((prev) => ({ ...prev, [key]: value }));
    };
  }

  async function rerankExistingResults(currentWeights = weights) {
    if (!searchContext?.searchId) {
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const data = await fetchJson("/api/recommend/rerank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          search_id: searchContext.searchId,
          weights: {
            popularity: currentWeights.popularity,
            quality: currentWeights.quality,
            age: currentWeights.age,
            similarity: currentWeights.similarity,
          },
        }),
      });

      setGames(data.results || []);
      setMessage(data.message || "");
      setSearchContext((prev) => (
        prev ? { ...prev, searchId: data.search_id || prev.searchId } : prev
      ));
    } catch (err) {
      setError(err.message || "Failed to rerank recommendations.");
    } finally {
      setLoading(false);
    }
  }

  function handleWeightRelease() {
    if (page !== "results" || !searchContext || loading) {
      return;
    }
    rerankExistingResults(weights);
  }

  function goHome() {
    setPage("home");
    setGames([]);
    setInputValue("");
    setSearchContext(null);
    setSelectedGame(null);
    setGameSuggestions([]);
    setActiveTags(new Set());
    setMessage("");
    setError("");
    setLoading(false);
  }

  if (page === "home") {
    return (
      <>
        <TopBar />

        <div className="hero">
          <div>
            <div className="hero-title">SteamRec</div>
            <div className="hero-sub">User and Game Based Steam Recommendations</div>
            {/* <div className="hero-sub-sub">Current database is up to 2022. Will be updated in the next version.</div> */}
          </div>

          <div className="search-wrap">
            <div className="mode-toggle">
              <button className={`mode-btn ${mode === "game" ? "active" : ""}`} onClick={() => handleModeChange("game")}>
                Based on Game
              </button>
              <button className={`mode-btn ${mode === "user" ? "active" : ""}`} onClick={() => handleModeChange("user")}>
                Based on User
              </button>
            </div>

            <div className="search-row">
              <input
                className="search-input"
                type="text"
                placeholder={mode === "user" ? "Enter Steam 64-bit user ID" : "Type a game name"}
                value={inputValue}
                onChange={(event) => handleInputChange(event.target.value)}
                onKeyDown={handleKey}
                autoFocus
              />
              <button className="search-btn" onClick={handleSearch}>
                Search
              </button>
            </div>

            {mode === "game" && (gameSuggestions.length > 0 || gameSearchLoading) && (
              <div className="suggestions-panel">
                {gameSearchLoading ? (
                  <div className="suggestions-status">Searching games...</div>
                ) : (
                  gameSuggestions.map((game) => (
                    <GameSuggestion key={game.id} game={game} onSelect={runGameSearch} />
                  ))
                )}
              </div>
            )}

            {/* <div className={`status-message${error ? " error-message" : ""}`}>
              <span>{mode === "game" ? "Type game name and click Search to see the options." : "Enter Steam 64-bit user ID."}</span>
            </div>*/}
            {/* <div className={`status-message${error ? " error-message" : ""} status-message-sub`}>
              <span>This will be updated in future versions.</span>
            </div>*/}
          </div>
        </div>
      </>
    );
  }

  if (page === "loading") {
    return (
      <>
        <TopBar />

        <div className="loading-wrap">
          <div className="spinner" />
          <div className="loading-text">COMPUTING RECOMMENDATIONS...</div>
        </div>
      </>
    );
  }

  return (
    <>
      <TopBar onHomeClick={goHome} showHomeButton />

      <div className="results-layout">
        <div className="panel-left">
          <div className="panel-header">
            <span className="panel-header-title">
              {searchContext?.mode === "game" ? "Game Recommendations" : "User Recommendations"}
            </span>
            <span className="panel-header-id">
              {searchContext?.mode === "game"
                ? `${searchContext?.label || "Game"} (${searchContext?.id || ""})`
                : `ID: ${searchContext?.id || ""}`}
            </span>
            <span className="panel-header-count">{visibleGames.length} results</span>
          </div>

          <div className="cards-scroll">
            {error ? (
              <div className="empty-state">{error}</div>
            ) : message && visibleGames.length === 0 ? (
              <div className="empty-state">{message}</div>
            ) : visibleGames.length === 0 ? (
              <div className="empty-state">No games match the selected tag filters.</div>
            ) : (
              visibleGames.map((game) => (
                <GameCard key={game.id} game={game} />
              ))
            )}
          </div>
        </div>

        <div className="panel-right">
          <div className="weights-section">
            <div className="section-title">Blend Weights</div>
            {/* <div className="section-copy">Recommendations refresh when you release a slider.</div> */}

            <SliderRow
              label="Popularity"
              desc="Higher values will prefer games that are more popular based on user reviews."
              value={weights.popularity}
              onChange={setWeight("popularity")}
              onRelease={handleWeightRelease}
            />
            <SliderRow
              label="Quality"
              desc="Higher values will prefer games with better ratings."
              value={weights.quality}
              onChange={setWeight("quality")}
              onRelease={handleWeightRelease}
            />
            <SliderRow
              label="Recency"
              desc="Higher values will prefer newer games."
              value={weights.age}
              onChange={setWeight("age")}
              onRelease={handleWeightRelease}
            />
            <SliderRow
              label="Similarity"
              desc="Higher values will prefer games more similar in tags of the selected game."
              value={weights.similarity}
              onChange={setWeight("similarity")}
              onRelease={handleWeightRelease}
            />
          </div>

          <div className="tags-section">
            <div className="tags-control">
              <div className="section-title" style={{ marginBottom: 0 }}>Filter by Tag</div>
              {activeTags.size > 0 && <button className="reset-btn" onClick={resetTags}>reset</button>}
            </div>

            <div className="tags-list">
              {sortedTags.map((tag) => (
                <span
                  key={tag}
                  className={`filter-tag${activeTags.has(tag) ? " active" : ""}`}
                  onClick={() => toggleTag(tag)}
                >
                  {tag}
                  <span className="count">{allTagCounts[tag]}</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
