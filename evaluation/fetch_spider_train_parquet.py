"""Download source-identified publisher xlangai/spider TRAIN parquet only (no DEV)."""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.report.exists():
        raise FileExistsError("no overwrite of frozen independent training source")
    import pandas as pd
    endpoint="https://datasets-server.huggingface.co/parquet?dataset=xlangai/spider"
    with urlopen(Request(endpoint,headers={"User-Agent":"source-verified-research/1.0"}),timeout=90) as resp:
        meta=json.load(resp)
    parts=sorted((f for f in meta["parquet_files"]
                  if f["config"]=="spider" and f["split"]=="train"),
                 key=lambda f:f["filename"])
    if not 1<=len(parts)<=10:
        raise ValueError("Spider TRAIN parquet publisher files not found")
    frames,identities=[],[]
    for f in parts:
        url=f["url"]
        if not url.startswith("https://huggingface.co/datasets/xlangai/spider/"):
            raise ValueError("unexpected publisher file origin")
        with urlopen(url,timeout=180) as resp:
            raw=resp.read(50_000_001)
        if len(raw)>50_000_000:
            raise ValueError("unbounded Spider source part")
        frame=pd.read_parquet(io.BytesIO(raw))
        frames.append(frame)
        identities.append({"filename":f["filename"],"sha256":sha256(raw).hexdigest(),"rows":len(frame)})
    data=pd.concat(frames,ignore_index=True)
    if not {"question","query","db_id"}.issubset(data.columns) or len(data)<6000:
        raise ValueError("missing original Spider TRAIN question/SQL source")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    data.to_parquet(a.output,index=False)
    a.report.write_text(json.dumps({
        "original_publisher":"xlangai/spider",
        "split":"train","license":"cc-by-sa-4.0",
        "source_parts":identities,"total_rows":len(data),
        "local_parquet_sha256":sha256(a.output.read_bytes()).hexdigest(),
        "no_DEV_source":True,"no_gold_BIRD_used":True,
    },indent=2,sort_keys=True)+"\n")
    print("SPIDER_SOURCE_TRAIN_PINNED",json.dumps({
        "parts":len(parts),"rows":len(data),
        "sha256":sha256(a.output.read_bytes()).hexdigest(),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
