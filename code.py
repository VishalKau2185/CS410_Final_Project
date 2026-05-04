import pandas as pd
import string

df = pd.read_csv("mpst_full_data.csv")


# function to clean text
def clean(plot):

    # check if text is empty
    if pd.isna(plot):
        return ""
    
    # convert to lowercase
    plot = plot.lower()
    
    # remove punctuation
    plot = plot.translate(str.maketrans('', '', string.punctuation))
    
    return plot


df["clean_plot"] = df["plot_synopsis"].apply(clean)


# Before
print("Before:")
print(df["plot_synopsis"][0])

# After
print("After:")
print(df["clean_plot"][0])


df["plot_length"] = df["plot_synopsis"].apply(lambda x: len(str(x).split()))

# show stats
print(df["plot_length"].describe())


df["original"] = df["plot_synopsis"].str[:20]
df["clean"] = df["clean_plot"].str[:20]

print(df[["original", "clean"]].head(5).to_string(index=False))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = df[["title", "plot_synopsis", "clean_plot"]]
df = df.dropna()


# do tf-idf for the plots
v = TfidfVectorizer(stop_words="english")
m = v.fit_transform(df["clean_plot"])


def search(q):

    q = clean(q)

    qv = v.transform([q])

    scores = cosine_similarity(qv, m).flatten()

    order = scores.argsort()[::-1]

    print("TF-IDF results for:", q)
    print()

    for i in range(5):
        a = order[i]

        print(i + 1, "-", df["title"].iloc[a])
        print("Score:", round(scores[a], 4))
        print("Plot:", df["plot_synopsis"].iloc[a][:250])
        print()


# baseline to compare
def baseline(q):

    q = clean(q)
    words = q.split()

    scores = []

    for plot in df["clean_plot"]:
        count = 0

        for w in words:
            if w in plot:
                count = count + 1

        scores.append(count)

    df["base_score"] = scores

    temp = df.sort_values(by="base_score", ascending=False)

    print("Baseline results for:", q)
    print()

    for i in range(5):
        print(i + 1, "-", temp["title"].iloc[i])
        print("Score:", temp["base_score"].iloc[i])
        print("Plot:", temp["plot_synopsis"].iloc[i][:250])
        print()


# test queries
qs = [
    "funny romantic comedy",
    "soldier returns home after war",
    "missing person investigation"
]


for q in qs:
    print("====================================")
    search(q)

    print("------------------------------------")
    baseline(q)
    print("====================================")
    print()