import project



if __name__ == "__main__":
    engine = project.Engine("mpst_full_data.csv", 300, "title", "plot_synopsis")
    engine.preprocess()
    engine.show_stats()

    # test queries
    qs = [
        "funny romantic comedy",
        "soldier returns home after war",
        "missing person investigation"
    ]

    for q in qs:
        print(f"==================================== QUERY: {q} ====================================")
        engine.search(q)

        print("------------------BASELINE------------------")
        engine.baseline(q)

        print("====================================")
        print()
