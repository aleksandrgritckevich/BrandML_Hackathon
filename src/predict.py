import torch
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_data(path):
    df = pd.read_json(path, lines=True)

    post_ids = df[df["root_id"].isna()]["id"].values
    comments = df[df["parent_id"].isin(post_ids)]
    posts = df[df["id"].isin(post_ids)]

    posts_comments = posts.merge(comments, left_on="id", right_on="root_id"). \
        groupby(["id_x", "text_x", "hash_x"])["hash_y"].apply(list).reset_index()[["text_x", "hash_x"]]
    hashes = posts.merge(comments, left_on="id", right_on="root_id"). \
        groupby(["id_x", "text_x", "hash_x"])["hash_y"].apply(list).reset_index()["hash_y"]
    all_comments = posts.merge(comments, left_on="id", right_on="root_id"). \
        groupby(["id_x", "text_x", "hash_x"])["text_y"].apply(list).reset_index()["text_y"]
    all_comments = all_comments.apply(lambda x: list(map(lambda y: y.replace("\n", ". "), x)))
    posts_comments["comments_hash"] = hashes
    posts_comments["comments"] = all_comments

    posts_comments.columns = ["text", "hash", "comments_hash", "comments"]
    posts_comments["comments"] = posts_comments["comments"].apply(lambda x: " \n".join(x))

    return posts_comments


def filter_comments(posts_comments,model,min_threshold,max_threshold):
    new_comments = []
    new_hashes = []
    for i in range(posts_comments.shape[0]):
        post = posts_comments.iloc[i]["text"]
        comments = posts_comments.iloc[i]["comments"].split("\n")
        hashes = posts_comments.iloc[i]["comments_hash"]
        post_embedding = torch.tensor(model.encode(post))
        comments_embeddings = torch.tensor(model.encode(comments))
        similarities = []
        similarity = torch.cosine_similarity(post_embedding.unsqueeze(0), comments_embeddings)
        index = (similarity > min_threshold) & (similarity < max_threshold)
        if len(comments) == 0:
            comments = [""]
        elif len(comments) == 1:
            if index[0].item() != True:
                comments = [""]
        else:
            comments = "\n".join(np.array(comments)[index])
            hashes = list(np.array(hashes)[index])
        new_comments.append(comments)
        new_hashes.append(hashes)

    posts_comments["comments"] = new_comments
    posts_comments["comments_hash"] = new_hashes

    features = []
    for i in range(len(posts_comments)):
        comments = posts_comments['comments'].iloc[i]
        post = posts_comments['text'].iloc[i]
        hash = posts_comments["hash"].iloc[i]
        comments_hash = posts_comments["comments_hash"].iloc[i]
        feature = f"Пост:\n\n{post}\n\nКомментарии:\n\n{comments}\n\n Текст выше содержит комментарии к посту в соцсети. Опиши общий смысл этих комментариев одним абзацем."
        features.append((feature, hash, comments_hash))

    return features


# Предсказываем саммари на нашем датасете.

def predict(features,model,tokenizer):
    predict = []
    hashes = []
    comments_hashes = []
    with torch.no_grad():
        for post, hash, comments_hash in features:
            inputs = torch.tensor(tokenizer(post, truncation=True, max_length=500)["input_ids"])
            predict.append(tokenizer.decode(model.generate(input_ids=inputs.unsqueeze(0).to(device), max_length=100)[0],
                                            skip_special_tokens=True))
            hashes.append(hash)
            comments_hashes.append(comments_hash)
    return predict,hashes,comments_hashes
