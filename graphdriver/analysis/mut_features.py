import torch
from graphdriver.commons import data


def get(cancer: str):
    cm_data = data.Dataset(cancer=cancer).get_data()
    labels_specific = cm_data.labels.drivers_cancer
    features_specific = cm_data.x[labels_specific, :]
    avg_specific, std_specific = torch.mean(features_specific, axis=0), torch.std(features_specific, axis=0)
    assert avg_specific.shape[0] == features_specific.shape[1], ValueError("avg not correct")

    labels_others = None
    labels_others = cm_data.labels.drivers_others
    labels_others = torch.cat((labels_others, cm_data.labels.candidates))
    features_others = cm_data.x[labels_others, :]
    avg_others, std_others = torch.mean(features_others, axis=0), torch.std(features_others, axis=0)
    assert avg_others.shape[0] == features_others.shape[1], ValueError("avg not correct")

    labels_passengers = cm_data.labels.passengers
    features_passengers = cm_data.x[labels_passengers, :]
    avg_passengers, std_passengers = torch.mean(features_passengers, axis=0), torch.std(features_passengers, axis=0)
    assert avg_passengers.shape[0] == features_passengers.shape[1], ValueError("avg not correct")
    return [avg_specific, std_specific], [avg_others, std_others], [avg_passengers, std_passengers]


def utils():
    df = pd.DataFrame([avg_specific.numpy(), avg_passengers.numpy()]).T
    print(df)
    df.std(axis=1)
    # pd.DataFrame([avg_specific.numpy(), std_specific.numpy(), avg_others.numpy(), std_others.numpy(), avg_passengers.numpy(), std_passengers.numpy()]).T
