import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

print('-'*10)
print('MOVIE RECOMMENDATIONS')
print('-'*10)

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# warp -> weighted approximate rank pairwise
model = LightFM(loss='warp')

model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):
    # no of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:
        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our models predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in the order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print(" " * 4, "Known positives:")

        for x in known_positives[:3]:
            print(" " * 10, "%s" % x)

        print(" " * 4, "Recommended:")

        for x in top_items[:3]:
            print(" " * 10, x)


sample_recommendation(model, data, [3, 25, 450])
