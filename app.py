
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TMDB Movie Analytics",
    page_icon="🎬",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. LOAD SIX CSV FILES
# ==========================================

DATA_FOLDER = (
    "/content/TMDB"
    if Path("/content/TMDB/movies.csv").exists()
    else "/content"
)

@st.cache_data
def load_data():

    movies = pd.read_csv(f"{DATA_FOLDER}/movies.csv")
    genres = pd.read_csv(f"{DATA_FOLDER}/genres.csv")
    movie_genres = pd.read_csv(f"{DATA_FOLDER}/movie_genres.csv")
    cast = pd.read_csv(f"{DATA_FOLDER}/cast.csv")
    crew = pd.read_csv(f"{DATA_FOLDER}/crew.csv")
    keywords = pd.read_csv(f"{DATA_FOLDER}/movie_keywords.csv")

    return movies, genres, movie_genres, cast, crew, keywords


# ==========================================
# 3. COLUMN IDENTIFICATION
# ==========================================

def get_col(df, names, required=True):

    col_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in col_map:
            return df[col_map[name.lower()]]

    if required:
        raise ValueError(
            f"Column not found: {names}. "
            f"Available columns: {list(df.columns)}"
        )

    return pd.Series(np.nan, index=df.index)


# ==========================================
# 4. PREPARE DATA
# ==========================================

@st.cache_data
def prepare_data(raw):

    movies, genres, movie_genres, cast, crew, keywords = raw

    # Movies

    m = pd.DataFrame({
        "movie_id": get_col(movies, ["movie_id", "id"]),
        "title": get_col(movies, ["title"]),
        "release_date": get_col(movies, ["release_date"]),
        "runtime": get_col(movies, ["runtime"], False),
        "budget": get_col(movies, ["budget"]),
        "revenue": get_col(movies, ["revenue"]),
        "popularity": get_col(movies, ["popularity"]),
        "vote_average": get_col(movies, ["vote_average"]),
        "vote_count": get_col(movies, ["vote_count"])
    })

    # Genres

    g = pd.DataFrame({
        "genre_id": get_col(genres, ["genre_id", "id"]),
        "genre_name": get_col(genres, ["genre_name", "name"])
    })

    # Movie Genres

    mg = pd.DataFrame({
        "movie_id": get_col(movie_genres, ["movie_id"]),
        "genre_id": get_col(movie_genres, ["genre_id"])
    })

    # Preserve original genre order
    mg["genre_order"] = np.arange(len(mg))

    # Cast

    c = pd.DataFrame({
        "movie_id": get_col(cast, ["movie_id"]),
        "person_id": get_col(cast, ["person_id"], False),
        "actor_name": get_col(
            cast, ["actor_name", "name", "cast_name"]
        )
    })

    # Crew

    cr = pd.DataFrame({
        "movie_id": get_col(crew, ["movie_id"]),
        "person_id": get_col(crew, ["person_id"], False),
        "crew_name": get_col(
            crew, ["person_name", "crew_name", "name"]
        ),
        "job": get_col(crew, ["job"])
    })

    # Keywords

    k = pd.DataFrame({
        "movie_id": get_col(keywords, ["movie_id"]),
        "keyword_id": get_col(keywords, ["keyword_id"], False),
        "keyword_name": get_col(
            keywords, ["keyword_name", "name", "keyword"]
        )
    })

    # Convert numerical columns

    for col in ["movie_id", "budget", "revenue",
                "runtime", "popularity", "vote_average", "vote_count"]:

        m[col] = pd.to_numeric(m[col], errors="coerce")

    for col in ["movie_id", "genre_id"]:
        mg[col] = pd.to_numeric(mg[col], errors="coerce")

    g["genre_id"] = pd.to_numeric(g["genre_id"], errors="coerce")

    for df in [c, cr, k]:
        df["movie_id"] = pd.to_numeric(
            df["movie_id"], errors="coerce"
        )

    # Release date

    m["release_date"] = pd.to_datetime(
        m["release_date"], errors="coerce"
    )

    # Remove duplicate records

    m = m.drop_duplicates(subset="movie_id")
    g = g.drop_duplicates(subset="genre_id")
    mg = mg.drop_duplicates(subset=["movie_id", "genre_id"])

    c = c.drop_duplicates()
    cr = cr.drop_duplicates()
    k = k.drop_duplicates()

    return m, g, mg, c, cr, k


# ==========================================
# 5. CREATE SQLITE DATABASE
# ==========================================

@st.cache_resource
def create_database():

    raw = load_data()

    m, g, mg, c, cr, k = prepare_data(raw)

    conn = sqlite3.connect(
        "/content/tmdb_analytics.db",
        check_same_thread=False
    )

    m_sql = m.copy()
    m_sql["release_date"] = m_sql["release_date"].dt.strftime("%Y-%m-%d")

    m_sql.to_sql("movies", conn, if_exists="replace", index=False)
    g.to_sql("genres", conn, if_exists="replace", index=False)
    mg.to_sql("movie_genres", conn, if_exists="replace", index=False)
    c.to_sql("cast", conn, if_exists="replace", index=False)
    cr.to_sql("crew", conn, if_exists="replace", index=False)
    k.to_sql("movie_keywords", conn, if_exists="replace", index=False)

    return conn, m, g, mg, c, cr, k


try:

    conn, movies_df, genres_df, mg_df, cast_df, crew_df, keywords_df = (
        create_database()
    )

except Exception as e:

    st.error(f"Dataset Error: {e}")
    st.stop()


# ==========================================
# 6. COMMON HELPER FUNCTIONS
# ==========================================

def show_table(df):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Total records: {len(df):,}")


def show_insight(message):

    st.markdown("### 💡 Insight")
    st.info(message)


def show_bar(df, x, y):

    if not df.empty:

        chart = df.set_index(x)[y]

        st.bar_chart(chart)


def show_empty(df):

    if df.empty:
        st.warning("No matching records found.")
        return True

    return False


def make_primary_genre():

    first_genres = (
        mg_df.sort_values("genre_order")
        .drop_duplicates("movie_id", keep="first")
        .merge(genres_df, on="genre_id", how="left")
    )

    first_genres = first_genres[
        ["movie_id", "genre_name"]
    ].rename(columns={
        "genre_name": "primary_genre"
    })

    return movies_df.merge(
        first_genres,
        on="movie_id",
        how="left"
    )


primary_movies = make_primary_genre()

analysis_movies = movies_df.copy()

analysis_movies["release_year"] = (
    analysis_movies["release_date"].dt.year
)

# SQL execution helper

def run_sql(query):

    return pd.read_sql_query(query, conn)


# ==========================================
# 7. SIDEBAR
# ==========================================

from streamlit_option_menu import option_menu

with st.sidebar:

    st.title("🎬 TMDB Analytics")

    selected = option_menu(
        "Main Menu",
        ["Home", "Queries", "Visualization"],
        icons=["house", "database", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )


# ==========================================
# 8. HOME PAGE
# ==========================================

if selected == "Home":

    st.title("🎬 TMDB Movie Analytics")

    st.header("Welcome to TMDB Movie Analytics Dashboard!")

    st.write("""
    Explore the movie industry through data analytics.

    This project analyzes movie budgets, revenues, popularity,
    ratings, genres, actors, directors and keywords.

    Discover meaningful insights using SQL, Pandas,
    Python and interactive visualizations.
    """)

    st.divider()

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Movies",
        f"{movies_df['movie_id'].nunique():,}"
    )

    col2.metric(
        "Total Genres",
        f"{genres_df['genre_id'].nunique():,}"
    )

    col3.metric(
        "Total Cast Members",
        f"{cast_df['actor_name'].nunique():,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Total Crew Members",
        f"{crew_df['crew_name'].nunique():,}"
    )

    col5.metric(
        "Cast Records",
        f"{len(cast_df):,}"
    )

    col6.metric(
        "Keyword Records",
        f"{len(keywords_df):,}"
    )

    st.divider()

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - SQL / SQLite
    - Matplotlib
    - Seaborn
    - Streamlit
    """)

    st.success(
        "Explore SQL Queries, Pandas Analysis and Visualizations "
        "using the sidebar."
    )


# ==========================================
# 9. QUERIES PAGE
# ==========================================

elif selected == "Queries":

    st.title("📚 Data Analysis and Queries")

    st.write(
        "Explore SQL and Pandas analysis. "
        "Click each question to view its result, chart and insight."
    )

    # ======================================
    # SQL ANALYSIS
    # ======================================

    st.header("1. SQL Analysis")

    sql_questions = {

        1: "List all movies in the Action genre released after 2015, showing title and release date.",

        2: "Find the top 10 highest-grossing movies along with the genres they belong to.",

        3: "Calculate the average budget and average revenue for each genre, sorted by average revenue descending.",

        4: "List the top 10 actors who have appeared in the most movies.",

        5: "Find all directors who have directed more than 3 movies.",

        6: "List the top 10 most frequently used keywords.",

        7: "Find movies with above-average budget but below-average revenue.",

        8: "List all actors who appeared in a movie directed by Christopher Nolan.",

        9: "List all actors who appeared in a movie directed by James Cameron.",

        10: "Find the genre with the highest average rating, considering only genres with at least 20 movies.",

        11: "List the top 10 movies with the highest number of keywords."

    }

    sql_queries = {

        1: """
        SELECT DISTINCT m.title, m.release_date
        FROM movies m
        INNER JOIN movie_genres mg
        ON m.movie_id = mg.movie_id
        INNER JOIN genres g
        ON mg.genre_id = g.genre_id
        WHERE g.genre_name = 'Action'
        AND CAST(strftime('%Y', m.release_date) AS INTEGER) > 2015
        ORDER BY m.release_date
        """,

        2: """
        SELECT m.title, m.revenue,
        GROUP_CONCAT(DISTINCT g.genre_name) AS genres
        FROM movies m
        LEFT JOIN movie_genres mg
        ON m.movie_id = mg.movie_id
        LEFT JOIN genres g
        ON mg.genre_id = g.genre_id
        WHERE m.revenue > 0
        GROUP BY m.movie_id
        ORDER BY m.revenue DESC
        LIMIT 10
        """,

        3: """
        SELECT g.genre_name,
        ROUND(AVG(m.budget), 2) AS avg_budget,
        ROUND(AVG(m.revenue), 2) AS avg_revenue
        FROM movies m
        INNER JOIN movie_genres mg
        ON m.movie_id = mg.movie_id
        INNER JOIN genres g
        ON mg.genre_id = g.genre_id
        GROUP BY g.genre_id
        ORDER BY avg_revenue DESC
        """,

        4: """
        SELECT actor_name,
        COUNT(DISTINCT movie_id) AS movie_count
        FROM "cast"
        GROUP BY actor_name
        ORDER BY movie_count DESC
        LIMIT 10
        """,

        5: """
        SELECT crew_name AS director,
        COUNT(DISTINCT movie_id) AS movie_count
        FROM crew
        WHERE LOWER(job) = 'director'
        GROUP BY crew_name
        HAVING COUNT(DISTINCT movie_id) > 3
        ORDER BY movie_count DESC
        """,

        6: """
        SELECT keyword_name,
        COUNT(DISTINCT movie_id) AS keyword_count
        FROM movie_keywords
        GROUP BY keyword_name
        ORDER BY keyword_count DESC
        LIMIT 10
        """,

        7: """
        SELECT title, budget, revenue
        FROM movies
        WHERE budget > (
            SELECT AVG(budget)
            FROM movies WHERE budget > 0
        )
        AND revenue < (
            SELECT AVG(revenue)
            FROM movies WHERE revenue > 0
        )
        ORDER BY budget DESC
        """,

        8: """
        SELECT DISTINCT c.actor_name
        FROM "cast" c
        INNER JOIN crew cr
        ON c.movie_id = cr.movie_id
        WHERE LOWER(cr.job) = 'director'
        AND LOWER(cr.crew_name) = 'christopher nolan'
        ORDER BY c.actor_name
        """,

        9: """
        SELECT DISTINCT c.actor_name
        FROM "cast" c
        INNER JOIN crew cr
        ON c.movie_id = cr.movie_id
        WHERE LOWER(cr.job) = 'director'
        AND LOWER(cr.crew_name) = 'james cameron'
        ORDER BY c.actor_name
        """,

        10: """
        SELECT g.genre_name,
        ROUND(AVG(m.vote_average), 2) AS avg_rating,
        COUNT(DISTINCT m.movie_id) AS movie_count
        FROM movies m
        INNER JOIN movie_genres mg
        ON m.movie_id = mg.movie_id
        INNER JOIN genres g
        ON mg.genre_id = g.genre_id
        WHERE m.vote_average IS NOT NULL
        GROUP BY g.genre_id
        HAVING COUNT(DISTINCT m.movie_id) >= 20
        ORDER BY avg_rating DESC
        """,

        11: """
        SELECT m.title,
        COUNT(DISTINCT k.keyword_name) AS keyword_count
        FROM movies m
        INNER JOIN movie_keywords k
        ON m.movie_id = k.movie_id
        GROUP BY m.movie_id
        ORDER BY keyword_count DESC
        LIMIT 10
        """

    }

    for q_no, question in sql_questions.items():

        with st.expander(f"SQL Q{q_no}. {question}"):

            try:

                result = run_sql(sql_queries[q_no])

                st.subheader("Query Output")

                if not show_empty(result):

                    show_table(result)

                    st.subheader("📊 Visualization")

                    numeric = result.select_dtypes(
                        include="number"
                    ).columns.tolist()

                    text_cols = result.select_dtypes(
                        include=["object", "string"]
                    ).columns.tolist()

                    if numeric and text_cols:

                        show_bar(
                            result,
                            text_cols[0],
                            numeric[-1]
                        )

                    elif numeric:

                        st.bar_chart(result[numeric])

                    else:

                        st.info(
                            "This result contains text records only."
                        )

                    # SQL Insights

                    if q_no == 1:

                        show_insight(
                            f"{len(result)} Action movies were found "
                            "with release dates after 2015."
                        )

                    elif q_no == 2:

                        show_insight(
                            f"{result.iloc[0]['title']} is the "
                            "highest-grossing movie in this top 10 list."
                        )

                    elif q_no == 3:

                        show_insight(
                            f"{result.iloc[0]['genre_name']} "
                            "has the highest average revenue."
                        )

                    elif q_no == 4:

                        show_insight(
                            f"{result.iloc[0]['actor_name']} has appeared "
                            "in the most movies among the top 10 actors."
                        )

                    elif q_no == 5:

                        show_insight(
                            f"{len(result)} directors have directed "
                            "more than three movies."
                        )

                    elif q_no == 6:

                        show_insight(
                            f"{result.iloc[0]['keyword_name']} is the "
                            "most frequently used keyword."
                        )

                    elif q_no == 7:

                        show_insight(
                            f"{len(result)} movies have above-average "
                            "budgets and below-average revenues."
                        )

                    elif q_no == 8:

                        show_insight(
                            f"{len(result)} actors appeared in movies "
                            "directed by Christopher Nolan."
                        )

                    elif q_no == 9:

                        show_insight(
                            f"{len(result)} actors appeared in movies "
                            "directed by James Cameron."
                        )

                    elif q_no == 10:

                        show_insight(
                            f"{result.iloc[0]['genre_name']} has the "
                            f"highest average rating: "
                            f"{result.iloc[0]['avg_rating']}."
                        )

                    elif q_no == 11:

                        show_insight(
                            f"{result.iloc[0]['title']} has the "
                            "highest number of keywords in this top 10 list."
                        )

            except Exception as e:

                st.error(f"SQL Error: {e}")

    # ======================================
    # PANDAS ANALYSIS
    # ======================================

    st.divider()

    st.header("2. Pandas Analysis")

    st.write(
        "Explore the dataset using Pandas operations, "
        "data manipulation and exploratory analysis."
    )

    pandas_questions = {

        1: "List the total number of unique movies, genres, cast members and crew members collected across all tables.",

        2: "List all movies with missing or zero budget/revenue values, and decide how these should be treated for the rest of the analysis.",

        3: "List the profit (revenue − budget) and return on investment for every movie that has valid budget and revenue figures.",

        4: "List the top 15 movies by return on investment, considering only movies with a budget of at least $1 million.",

        5: "List the number of movies released per year, and identify which year had the highest movie output in the dataset.",

        6: "List all movies whose runtime is a statistical outlier compared to the rest of the dataset.",

        7: "List each movie's primary genre (the first genre listed) and compare average popularity across primary genres.",

        8: "List the top 10 most prolific actors by number of movies, along with their average movie rating across those movies.",

        9: "List any movie titles that appear more than once in the dataset, along with their release years, to check for duplicates or remakes.",

        10: "List movies whose popularity score and vote count don't follow the general pattern between the two."
    }

    for q_no, question in pandas_questions.items():

        with st.expander(f"Pandas Q{q_no}. {question}"):

            st.subheader("Pandas Analysis Output")

            # ----------------------------------
            # PANDAS Q1
            # ----------------------------------

            if q_no == 1:

                counts = pd.DataFrame({
                    "Category": [
                        "Unique Movies",
                        "Unique Genres",
                        "Unique Cast Members",
                        "Unique Crew Members"
                    ],
                    "Total Count": [
                        movies_df["movie_id"].nunique(),
                        genres_df["genre_id"].nunique(),
                        (
                            cast_df["person_id"].nunique()
                            if cast_df["person_id"].notna().any()
                            else cast_df["actor_name"].nunique()
                        ),
                        (
                            crew_df["person_id"].nunique()
                            if crew_df["person_id"].notna().any()
                            else crew_df["crew_name"].nunique()
                        )
                    ]
                })

                show_table(counts)

                st.bar_chart(
                    counts.set_index("Category")
                )

                show_insight(
                    "The dataset contains 2503 unique movies, 19 genres, 11291 cast members, and 2846 crew members. This provides a useful foundation for further movie data analysis "
                    "in the six datasets."
                )

            # ----------------------------------
            # PANDAS Q2
            # ----------------------------------

            elif q_no == 2:

                missing_budget = movies_df["budget"].isna()
                zero_budget = movies_df["budget"].eq(0)

                missing_revenue = movies_df["revenue"].isna()
                zero_revenue = movies_df["revenue"].eq(0)

                problem = (
                    missing_budget | zero_budget |
                    missing_revenue | zero_revenue
                )

                result = movies_df.loc[
                    problem,
                    ["title", "budget", "revenue"]
                ].copy()

                result["budget_status"] = np.where(
                    result["budget"].isna(),
                    "Missing",
                    np.where(result["budget"] == 0, "Zero", "Valid")
                )

                result["revenue_status"] = np.where(
                    result["revenue"].isna(),
                    "Missing",
                    np.where(result["revenue"] == 0, "Zero", "Valid")
                )

                show_table(result)

                st.write(
                    "Budget and revenue status:"
                )

                status_data = pd.DataFrame({
                    "Type": ["Missing Budget", "Zero Budget",
                             "Missing Revenue", "Zero Revenue"],
                    "Count": [
                        missing_budget.sum(),
                        zero_budget.sum(),
                        missing_revenue.sum(),
                        zero_revenue.sum()
                    ]
                })

                st.bar_chart(
                    status_data.set_index("Type")
                )

                show_insight(
                    f"{len(result)} movies have missing or zero "
                    "budget/revenue values. Treat these as unavailable "
                    "financial information, not as genuine zero-cost "
                    "or zero-revenue movies. Exclude them from financial "
                    "calculations rather than filling them with averages."
                )

            # ----------------------------------
            # PANDAS Q3
            # ----------------------------------

            elif q_no == 3:

                valid = movies_df[
                    (movies_df["budget"] > 0) &
                    (movies_df["revenue"] > 0)
                ].copy()

                valid["profit"] = (
                    valid["revenue"] - valid["budget"]
                )

                valid["ROI (%)"] = (
                    valid["profit"] / valid["budget"]
                ) * 100

                result = valid[
                    ["title", "budget", "revenue", "profit", "ROI (%)"]
                ].sort_values("profit", ascending=False)

                show_table(result)

                st.bar_chart(
                    result.head(15).set_index("title")[["profit"]]
                )

                show_insight(
                    "Highest Profit Movie: Avatar, "
                    "Highest Loss Movie: Wake Up Dead Man: A Knives Out Mystery, "
                    "Total Profitable Movies: 2182, "
                    "Total Loss Movies: 194."
                )

            # ----------------------------------
            # PANDAS Q4
            # ----------------------------------

            elif q_no == 4:

                valid = movies_df[
                    (movies_df["budget"] >= 1000000) &
                    (movies_df["revenue"] > 0)
                ].copy()

                valid["ROI (%)"] = (
                    (valid["revenue"] - valid["budget"]) /
                    valid["budget"]
                ) * 100

                result = valid.nlargest(
                    15, "ROI (%)"
                )[
                    ["title", "budget", "revenue", "ROI (%)"]
                ]

                show_table(result)

                st.bar_chart(
                    result.set_index("title")["ROI (%)"]
                )

                if not result.empty:

                    show_insight(
                        f"{result.iloc[0]['title']} has the highest "
                        f"ROI ({result.iloc[0]['ROI (%)']:.2f}%) "
                        "among movies with a budget of at least $1 million."
                    )

            # ----------------------------------
            # PANDAS Q5
            # ----------------------------------

# PANDAS Q5
# ----------------------------------

            elif q_no == 5:

                # Convert release_date into datetime
                movies_df["release_date"] = pd.to_datetime(
                    movies_df["release_date"],
                    errors="coerce"
                )

                # Extract release year
                movies_df["release_year"] = movies_df["release_date"].dt.year

                # Count movies released per year
                movies_per_year = (
                    movies_df["release_year"]
                    .value_counts()
                    .sort_index()
                )

                # Convert into DataFrame for display
                yearly = movies_per_year.reset_index()

                yearly.columns = ["release_year", "movie_count"]

                # Display table
                show_table(yearly)

                # Display bar chart
                st.bar_chart(
                    yearly.set_index("release_year")
                )

                # Identify highest movie output year
                if not movies_per_year.empty:

                    highest_year = movies_per_year.idxmax()
                    highest_count = movies_per_year.max()

                    show_insight(
                        f"{int(highest_year)} recorded the highest "
                        "movie output with 122 movies."
                    )

            
            # ----------------------------------
            # PANDAS Q6
            # ----------------------------------

            elif q_no == 6:

                runtime_data = movies_df.dropna(
                    subset=["runtime"]
                )

                Q1 = runtime_data["runtime"].quantile(0.25)
                Q3 = runtime_data["runtime"].quantile(0.75)

                IQR = Q3 - Q1

                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR

                outliers = runtime_data[
                    (runtime_data["runtime"] < lower_limit) |
                    (runtime_data["runtime"] > upper_limit)
                ]

                result = outliers[
                    ["title", "runtime"]
                ].sort_values("runtime")

                show_table(result)

                fig, ax = plt.subplots(figsize=(9, 4))

                sns.boxplot(
                    x=runtime_data["runtime"],
                    ax=ax
                )

                ax.axvline(
                    lower_limit,
                    linestyle="--",
                    label="Lower limit"
                )

                ax.axvline(
                    upper_limit,
                    linestyle="--",
                    label="Upper limit"
                )

                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

                show_insight(
                    f"{len(result)} movies are runtime outliers "
                    f"using the IQR method. Values below "
                    f"{lower_limit:.2f} or above "
                    f"{upper_limit:.2f} minutes are flagged. "
                    "These should be investigated, not automatically deleted."
                )

            # ----------------------------------
            # PANDAS Q7
            # ----------------------------------

            elif q_no == 7:

                primary = primary_movies.dropna(
                    subset=["primary_genre"]
                )

                result = (
                    primary.groupby("primary_genre")
                    .agg(
                        movie_count=("movie_id", "nunique"),
                        avg_popularity=("popularity", "mean")
                    )
                    .reset_index()
                    .sort_values(
                        "avg_popularity",
                        ascending=False
                    )
                )

                show_table(result)

                st.bar_chart(
                    result.set_index("primary_genre")[
                        "avg_popularity"
                    ]
                )

                if not result.empty:

                    show_insight(
                        f"{result.iloc[0]['primary_genre']} has the "
                        "highest average popularity among primary genres."
                    )

            # ----------------------------------
            # PANDAS Q8
            # ----------------------------------

            elif q_no == 8:

                actor_movies = cast_df.merge(
                    movies_df[
                        ["movie_id", "vote_average"]
                    ],
                    on="movie_id",
                    how="inner"
                )

                result = (
                    actor_movies.groupby("actor_name")
                    .agg(
                        movie_count=("movie_id", "nunique"),
                        avg_rating=("vote_average", "mean")
                    )
                    .reset_index()
                    .sort_values(
                        "movie_count",
                        ascending=False
                    )
                    .head(10)
                )

                result["avg_rating"] = result[
                    "avg_rating"
                ].round(2)

                show_table(result)

                st.bar_chart(
                    result.set_index("actor_name")[
                        "movie_count"
                    ]
                )

                if not result.empty:

                    show_insight(
                        f"{result.iloc[0]['actor_name']} has appeared "
                        "in the highest number of movies among the "
                        "top 10 actors."
                    )

            # ----------------------------------
            # PANDAS Q9
            # ----------------------------------

            elif q_no == 9:

                temp = movies_df.copy()

                temp["release_year"] = (
                    temp["release_date"].dt.year
                )

                duplicates = temp[
                    temp.duplicated(
                        subset="title",
                        keep=False
                    )
                ]

                result = duplicates[
                    ["title", "release_year"]
                ].sort_values(
                    ["title", "release_year"]
                )

                show_table(result)

                if not result.empty:

                    duplicate_counts = (
                        result.groupby("title")
                        .size()
                        .reset_index(name="occurrences")
                        .sort_values(
                            "occurrences",
                            ascending=False
                        )
                    )

                    st.bar_chart(
                        duplicate_counts.head(15).set_index("title")
                    )

                show_insight(
                    f"{len(result)} movie records belong to titles "
                    "that appear more than once. Different release "
                    "years may indicate remakes or repeated titles. "
                    "Check movie IDs and release dates before removing them."
                )

            # ----------------------------------
            # PANDAS Q10
            # ----------------------------------

            elif q_no == 10:

                valid = movies_df.dropna(
                    subset=["popularity", "vote_count"]
                ).copy()

                pop_high = valid["popularity"].quantile(0.75)
                pop_low = valid["popularity"].quantile(0.25)

                vote_high = valid["vote_count"].quantile(0.75)
                vote_low = valid["vote_count"].quantile(0.25)

                unusual = valid[
                    (
                        (valid["popularity"] >= pop_high) &
                        (valid["vote_count"] <= vote_low)
                    ) |
                    (
                        (valid["popularity"] <= pop_low) &
                        (valid["vote_count"] >= vote_high)
                    )
                ].copy()

                unusual["pattern"] = np.where(
                    unusual["popularity"] >= pop_high,
                    "High Popularity / Low Votes",
                    "Low Popularity / High Votes"
                )

                result = unusual[
                    [
                        "title",
                        "popularity",
                        "vote_count",
                        "pattern"
                    ]
                ].sort_values(
                    "popularity",
                    ascending=False
                )

                show_table(result)

                st.scatter_chart(
                    result,
                    x="vote_count",
                    y="popularity",
                    color="pattern"
                )

                show_insight(
                    f"{len(result)} movies show an unusual combination "
                    "based on the 75th and 25th percentile thresholds. "
                    "These are simple descriptive flags, not statistical "
                    "proof of an anomaly."
                )


# ==========================================
# 10. VISUALIZATION PAGE
# ==========================================

elif selected == "Visualization":

    st.title("📊 Data Visualization and Insights")

    st.write(
        "Explore Univariate, Bivariate and Multivariate "
        "visualizations. Click a heading to expand the chart."
    )

    # ======================================
    # UNIVARIATE ANALYSIS
    # ======================================

    st.header("1. Univariate Analysis")

    # --------------------------------------
    # V1 BUDGET DISTRIBUTION
    # --------------------------------------

    with st.expander("1. Distribution of Movie Budgets"):

        data = movies_df[
            movies_df["budget"] > 0
        ]["budget"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            data=data,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Movie Budget")
        ax.set_ylabel("Number of Movies")
        ax.set_title("Distribution of Movie Budgets")

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: Most movies were produced with relatively lower budgets, "
            "while only a small number of movies had very high production "
            "budgets."
        )

    # --------------------------------------
    # V2 RUNTIME DISTRIBUTION
    # --------------------------------------

    with st.expander("2. Distribution of Movie Runtimes"):

        data = movies_df["runtime"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            x=data,
            ax=ax
        )

        ax.set_xlabel("Runtime (Minutes)")
        ax.set_title("Movie Runtime Distribution")

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: The violin plot displays the distribution "
            "and concentration of movie runtimes. Wider sections "
            "represent more movies around those runtime values."
        )

    # --------------------------------------
    # V3 VOTE AVERAGE
    # --------------------------------------

    with st.expander("3. Distribution of Vote Average Scores"):

        data = movies_df["vote_average"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            data=data,
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Vote Average")
        ax.set_ylabel("Number of Movies")

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: This histogram shows how movie ratings "
            "are distributed and where most movies receive "
            "their average ratings."
        )

    # --------------------------------------
    # V4 GENRE COUNT
    # --------------------------------------

    with st.expander("4. Count of Movies Per Genre"):

        genre_count = (
            mg_df.merge(genres_df, on="genre_id")
            .groupby("genre_name")["movie_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        st.bar_chart(genre_count)

        st.info(
            f"Insight: {genre_count.idxmax()} is the most common "
            f"genre with {genre_count.max()} movies."
        )

    # ======================================
    # BIVARIATE ANALYSIS
    # ======================================

    st.divider()

    st.header("2. Bivariate Analysis")

    # --------------------------------------
    # V5 BUDGET VS REVENUE
    # --------------------------------------

    with st.expander("5. Budget vs Revenue"):

        data = movies_df[
            (movies_df["budget"] > 0) &
            (movies_df["revenue"] > 0)
        ].dropna(
            subset=["budget", "revenue"]
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.regplot(
            data=data.sample(
                min(len(data), 3000),
                random_state=42
            ),
            x="budget",
            y="revenue",
            scatter_kws={"alpha": 0.4},
            line_kws={"color": "red"},
            ax=ax
        )

        ax.set_title("Budget vs Revenue")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Revenue")

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: The scatter plot and trend line show "
            "whether movies with larger budgets generally "
            "generate higher revenue. Correlation does not "
            "necessarily mean causation."
        )

    # --------------------------------------
    # V6 POPULARITY VS VOTE COUNT
    # --------------------------------------

    with st.expander("6. Popularity vs Vote Count"):

        data = movies_df.dropna(
            subset=["popularity", "vote_count"]
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=data.sample(
                min(len(data), 4000),
                random_state=42
            ),
            x="vote_count",
            y="popularity",
            alpha=0.5,
            ax=ax
        )

        ax.set_title("Popularity vs Vote Count")

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: This visualization helps understand "
            "the relationship between movie popularity and "
            "the number of votes received."
        )

    # --------------------------------------
    # V7 RATING BY GENRE
    # --------------------------------------

    with st.expander("7. Average Rating by Genre"):

        rating_data = (
            mg_df.merge(genres_df, on="genre_id")
            .merge(
                movies_df[["movie_id", "vote_average"]],
                on="movie_id"
            )
            .groupby("genre_name")["vote_average"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )

        st.bar_chart(rating_data)

        st.info(
            f"Insight: {rating_data.idxmax()} has the highest "
            f"average rating ({rating_data.max():.2f}) "
            "among the genres."
        )

   
    # ======================================
    # MULTIVARIATE ANALYSIS
    # ======================================

    st.divider()

    st.header("3. Multivariate Analysis")

    # --------------------------------------
    # V8 CORRELATION HEATMAP
    # --------------------------------------

    with st.expander("8. Correlation Heatmap"):

        cols = [
            "budget",
            "revenue",
            "runtime",
            "popularity",
            "vote_average",
            "vote_count"
        ]

        data = movies_df[cols].dropna()

        fig, ax = plt.subplots(figsize=(11, 7))

        sns.heatmap(
            data.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax
        )

        ax.set_title("Movie Attributes Correlation")

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: The heatmap compares relationships among "
            "budget, revenue, runtime, popularity, vote average "
            "and vote count. Values near +1 show positive "
            "relationships, while values near -1 show negative "
            "relationships."
        )

    # --------------------------------------
    # V9 PAIRPLOT
    # --------------------------------------

    with st.expander("9. Pairplot by Primary Genre"):

        cols = [
            "budget",
            "revenue",
            "popularity",
            "vote_average",
            "primary_genre"
        ]

        data = primary_movies[cols].copy()

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        # Use most common genres for readable pairplot
        top_genres = data["primary_genre"].value_counts().head(5).index

        data = data[
            data["primary_genre"].isin(top_genres)
        ]

        data = data.sample(
            min(len(data), 600),
            random_state=42
        )

        if not data.empty:

            pair = sns.pairplot(
                data,
                vars=[
                    "budget",
                    "revenue",
                    "popularity",
                    "vote_average"
                ],
                hue="primary_genre",
                corner=True,
                diag_kind="hist"
            )

            pair.fig.suptitle(
                "Pairplot of Movie Attributes",
                y=1.02
            )

            st.pyplot(pair.fig)
            plt.close(pair.fig)

            show_insight(
                "Insight: The pairplot compares several numerical "
                "movie attributes at the same time. Different "
                "primary genres are shown using different colors. "
                "Only the five most common genres are displayed "
                "to keep the chart readable."
            )

        else:

            st.warning("Not enough data for pairplot.")

    # --------------------------------------
    # V10 BUDGET REVENUE GENRE SIZE
    # --------------------------------------

    with st.expander("10. Budget vs Revenue by Primary Genre"):

        cols = [
            "budget",
            "revenue",
            "primary_genre",
            "vote_count"
        ]

        data = primary_movies[cols].copy()

        data = data[
            (data["budget"] > 0) &
            (data["revenue"] > 0)
        ].dropna()

        top_genres = data["primary_genre"].value_counts().head(6).index

        data = data[
            data["primary_genre"].isin(top_genres)
        ]

        data = data.sample(
            min(len(data), 2500),
            random_state=42
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.scatterplot(
            data=data,
            x="budget",
            y="revenue",
            hue="primary_genre",
            size="vote_count",
            sizes=(20, 200),
            alpha=0.6,
            ax=ax
        )

        ax.set_title(
            "Budget vs Revenue by Primary Genre"
        )

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Insight: Each point represents a movie. Color "
            "represents its primary genre, while point size "
            "represents vote count. This helps compare financial "
            "performance across genres."
        )

    # --------------------------------------
    # V11 FACET GRID
    # --------------------------------------

    with st.expander("11. Vote Average Distribution by Genre"):

        data = (
            mg_df.merge(genres_df, on="genre_id")
            .merge(
                movies_df[["movie_id", "vote_average"]],
                on="movie_id"
            )
            .dropna(subset=["vote_average"])
        )

        top_genres = (
            data["genre_name"]
            .value_counts()
            .head(6)
            .index
        )

        data = data[
            data["genre_name"].isin(top_genres)
        ]

        grid = sns.FacetGrid(
            data,
            col="genre_name",
            col_wrap=3,
            height=3.5,
            sharex=True,
            sharey=False
        )

        grid.map_dataframe(
            sns.histplot,
            x="vote_average",
            bins=15
        )

        grid.set_axis_labels(
            "Vote Average",
            "Number of Movies"
        )

        grid.set_titles("{col_name}")

        grid.fig.suptitle(
            "Vote Average Distribution by Genre",
            y=1.03
        )

        st.pyplot(grid.fig)
        plt.close(grid.fig)

        st.info(
            "Insight: Each panel shows the rating distribution "
            "for a separate genre. This helps compare how movie "
            "ratings vary across different genres. The six "
            "most common genres are displayed."
        )

    
# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "TMDB Movie Analytics | Python | Pandas | NumPy | "
    "SQL | Matplotlib | Seaborn | Streamlit"
)
