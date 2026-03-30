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