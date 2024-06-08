from itertools import islice

from datasets import load_dataset


def is_cnn(record: dict) -> bool:
    return "(CNN)" in record.get("article", "")


def load_cnn_daily_mail(max_size: int = 0):
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").filter(is_cnn)
    articles = map(lambda row: row["article"], ds)
    if max_size > 0:
        articles = islice(articles, max_size)
    return list(articles)
