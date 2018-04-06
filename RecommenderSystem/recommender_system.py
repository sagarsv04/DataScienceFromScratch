import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Here are few good data sources https://gist.github.com/entaroadun/1653794

rating = 6.0
epoch = 40
user_ids = [3, 25, 450]

print_recommendations = True

# create model
model_warp = LightFM(loss='warp')
model_logistic = LightFM(loss='logistic')
model_bpr = LightFM(loss='bpr')
model_warp_kos = LightFM(loss='warp-kos')

# list of models with respective loss names
models = [["Warp",model_warp], ["Logistic",model_logistic], ["BPR", model_bpr], ["Warp-kos", model_warp_kos]]


def fetch_data(rating=4.0):
    # fetch data and format it
    print("Fetching Movie data with ratings: {0} and above.".format(rating))
    data = fetch_movielens(min_rating=rating)
    # len(data)
    return data


def split_train_test_data(data):
    # split training and testing data
    train_data = data['train']
    test_data = data['test']
    item_labels = data['item_labels']
    return train_data, test_data, item_labels


def train_models(model, model_name, train_data, epoch=30):
    # here we train our model
    # model is dynamic based on different model passed
    print("Traning model with loss: {0} for {1} epoch.".format(model_name, epoch))
    model.fit(train_data, epochs=epoch, num_threads=2)
    return model


def model_predict(model, user_id, n_items):
    # here we predict using our trained model
    # model is dynamic based on different classifiers passed
    scores = model.predict(user_id, np.arange(n_items))
    return scores


def run_recommendation(model, model_name, data, data_labels, user_ids):
    # model, model_name, data, data_labels, user_ids = model, model_name, train_data, item_labels, user_ids
    # number of users and movies in training data
    n_users, n_items = data.shape
    # generate recommendations for each user we input
    recomendations = {}
    for user_id in user_ids:
        # user_id = user_ids[0]
        user_recom = {}
        # movies they already like
        known_positives = data_labels[data.tocsr()[user_id].indices]
        # movies our model predicts they will like
        scores = model_predict(model, user_id, n_items)
        # rank them in order of most liked to least
        top_items = data_labels[np.argsort(-scores)]
        # store out the results
        user_recom["KnownPos"] = known_positives
        user_recom["ScoreAvg"] = np.mean(scores)
        user_recom["TopItems"] = top_items
        recomendations[user_id] = user_recom

    model_recomendation = [model_name,recomendations]

    return model_recomendation

def run_models():

    data = fetch_data()
    train_data, test_data, item_labels = split_train_test_data(data)

    # The best classifier list
    best_models = []
    for model in models:
        # model = models[0]
        model_name = model[0]
        model = model[1]
        model = train_models(model, model_name, train_data, epoch)
        # Testing using the same or different data
        model_recomendation = run_recommendation(model, model_name, train_data, item_labels, user_ids)
        best_models.append(model_recomendation)
    return best_models


def print_best_model(best_models):
    # To iterate over information use below info
    # best_models = [[model_name,{user_id:{KnownPos:array();ScoreAvg:num;TopItems:array()}}],... n models]

    for model_info in best_models:
        # model_info = best_models[0]
        print("Model Name: {0}".format(model_info[0]))

        for user_id in user_ids:
            # user_id = user_ids[0]
            print("User Id: {0}".format(user_id))
            print("Score Avg: {0}".format(model_info[1][user_id]["ScoreAvg"]))

            if print_recommendations:
                print("     Known positives:")
                for x in model_info[1][user_id]["KnownPos"][:3]:
                    print("        %s" % x)
                print("     Recommended:")
                for x in model_info[1][user_id]["TopItems"][:3]:
                    print("        %s" % x)

    return 0


def main():

    best_models = run_models()
    print_best_model(best_models)

    return 0


if __name__ == '__main__':
    main()
