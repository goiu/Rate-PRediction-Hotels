import pandas as pd
import random

if __name__ == "__main__":

    train_data = pd.DataFrame(columns=['train_reviews','train_ratings'])
    test_data = pd.DataFrame(columns=['test_reviews', 'test_ratings'])

    filepath = "./raw_data.csv"
    reading = pd.read_csv(filepath)

    temp_reviews = reading["reviews.title"]
    temp_ratings = reading["reviews.rating"]

    print(temp_reviews[0])
    print(temp_ratings[0])

    temp_combined = list(zip(temp_reviews, temp_ratings))
    random.shuffle(temp_combined)
    reviews, ratings = zip(*temp_combined)
    reviews = list(reviews)
    ratings = list(ratings)


    cutoff = int(0.8 * len(reviews))
    train_reviews = reviews[:cutoff]
    test_reviews = reviews[cutoff:]
    train_ratings = ratings[:cutoff]
    test_ratings = ratings[cutoff:]

    print("number of entries in training dataset: ", len(train_ratings))
    print("number of entries in testing dataset: ", len(test_ratings))

    train_data['train_reviews'] = train_reviews
    test_data['test_reviews'] = test_reviews
    train_data['train_ratings'] = train_ratings
    test_data['test_ratings'] = test_ratings

    train_data.to_csv('./train_data.csv', index=False)
    test_data.to_csv('./test_data.csv', index=False)



