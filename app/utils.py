from datetime import datetime
import os


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_parquet(df, path_prefix, timestamped=True, **kwargs):
    ts = f"_{timestamp()}" if timestamped else ""
    path = f"{path_prefix}{ts}.parquet"
    ensure_dir(path)
    df.to_parquet(path, **kwargs)
    return path


def save_csv(df, path_prefix, timestamped=True, **kwargs):
    ts = f"_{timestamp()}" if timestamped else ""
    path = f"{path_prefix}{ts}.csv"
    ensure_dir(path)
    df.to_csv(path, **kwargs)
    return path
