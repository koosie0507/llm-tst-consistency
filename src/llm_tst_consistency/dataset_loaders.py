from datasets import load_dataset


def is_cnn(record: dict) -> bool:
    return "(CNN)" in record.get("article", "")


def load_cnn_daily_mail():
    return load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").filter(is_cnn)
