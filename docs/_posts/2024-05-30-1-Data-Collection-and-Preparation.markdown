---
layout: post
title: "1. Data Collection and Preparation"
posted: "May 27, 2024"
categories: Super-Fast-LLM-Training
live: true
---
A language model learns everything it knows from the text we train it on. The model architecture, the optimiser and the training loop tricks in later posts only decide how fast and how well the model absorbs that text. If the text is full of navigation menus, cookie notices and the same page copied a hundred times, the model will learn to write navigation menus, cookie notices and repeat itself. So before we write any model code, we need good data.

In this series, we are building a small LLM called EduLLM that writes food recipes. In this post, we shall build the data pipeline for it. We will find recipe pages in Common Crawl, download only those pages, extract the recipe text, clean it, remove duplicates and save the result as JSONL files. The next post trains a tokeniser on these files.

LLM training happens in stages and each stage needs its own kind of data -

1. Pretraining. The model learns to predict the next token on a large amount of raw text.
2. Mid-training. Training continues on a smaller, higher quality mix (for example more code, maths or longer documents) to improve specific skills.
3. Post-training. The model learns to follow instructions from prompt and response pairs (supervised fine-tuning), and to prefer good answers over bad ones from preference data (RLHF or DPO).

This post covers only the pretraining data. We shall look at mid-training and post-training data in later posts.

## Where Pretraining Data Comes From

Most pretraining data comes from a small set of sources -

1. Web pages. This is by far the largest source and also the noisiest. Almost every open dataset starts from [Common Crawl](https://commoncrawl.org/).
2. Books. Long, well edited text. Public domain books from Project Gutenberg are a common choice.
3. Code. Public repositories from GitHub. Datasets like [The Stack](https://arxiv.org/abs/2211.15533) collect code with permissive licences.
4. Wikipedia. Small, but clean and factual.
5. Scientific papers. arXiv and PubMed Central.
6. Questions and answers. StackExchange and Reddit.

How much of each source to use is a design decision. GPT-3 (2020) sampled 60% of its training tokens from filtered Common Crawl, 22% from WebText2 (pages linked from Reddit), 16% from two book collections and 3% from Wikipedia. LLaMA (2023) used 67% Common Crawl, 15% C4, 4.5% each of GitHub, Wikipedia and books, 2.5% arXiv and 2% StackExchange. In both cases, more than two thirds of the data is web text. So most of the work on pretraining data is about cleaning web text.

## Open Pretraining Datasets

Let's look at how some well known open datasets turn Common Crawl into training data.

[C4](https://arxiv.org/abs/1910.10683) (Colossal Clean Crawled Corpus, 2019) was built for the T5 model. It takes the plain text (WET files, see below) of the April 2019 Common Crawl snapshot and applies a short list of rules. It keeps only lines that end with a punctuation mark, removes pages with bad words or code, keeps English pages and removes repeated three sentence spans. The result is about 750 GB of text. We shall implement these rules later in this post.

[CCNet](https://arxiv.org/abs/1911.00359) (Facebook, 2019) also starts from WET files. It removes duplicate paragraphs by hashing normalised paragraphs, identifies the language of each page with a fastText classifier, and scores every page with a small language model trained on Wikipedia. Pages with low perplexity (text that looks like Wikipedia) go into the "head" bucket, the rest go into "middle" and "tail". The Common Crawl part of LLaMA was built with CCNet.

[The Pile](https://arxiv.org/abs/2101.00027) (EleutherAI, 2020) is an 825 GiB mix of 22 sources, including PubMed Central, arXiv, GitHub, FreeLaw, StackExchange and books. For its web part (Pile-CC), the authors extracted text from the raw HTML in WARC files with jusText instead of using WET files, because the WET text kept too much boilerplate.

[RefinedWeb](https://arxiv.org/abs/2306.01116) (TII, 2023) was built for the Falcon models. It uses web data only. It extracts text from WARC files with trafilatura, drops pages from a URL block list, identifies the language with fastText, applies quality rules from the Gopher paper, and then removes duplicates in two steps. First it removes near duplicates with MinHash, and then exact repeated substrings. The paper showed that web data alone, if filtered and deduplicated well, can match curated mixes of books, papers and web pages. The full dataset has about 5 trillion tokens and 600 billion of them were released.

[RedPajama](https://www.together.ai/blog/redpajama) (Together, 2023) reproduced the LLaMA data mix in the open, about 1.2 trillion tokens. RedPajama-V2 (October 2023) is different. It processes 84 Common Crawl snapshots into 30 trillion deduplicated tokens and ships more than 40 quality signals with every document. It does not decide what to filter. The user chooses the thresholds.

[SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) (Cerebras, 2023) took the original RedPajama, removed very short documents and ran MinHash LSH near deduplication across all sources. 1.2 trillion tokens became 627 billion tokens. Almost half of the bytes were removed, nearly all of them as duplicates.

[Dolma](https://arxiv.org/abs/2402.00159) (AI2, 2023) was built for the OLMo models. It has 3 trillion tokens from Common Crawl, code, Reddit, scientific papers, Project Gutenberg books and Wikipedia. It uses a mix of rules from Gopher and C4, fastText language identification, and Bloom filters to remove duplicate URLs, documents and paragraphs. The tools used to build it are open source too.

Although the details differ, all of these pipelines follow the same steps -

1. Get the raw pages.
2. Extract the main text from the HTML.
3. Keep only the languages we want.
4. Remove low quality text with rules or classifiers.
5. Remove exact and near duplicates.

We shall follow the same steps, at a much smaller scale.

## Common Crawl

Common Crawl is a non-profit that has crawled the web since 2008 and publishes the data for free. A new snapshot is published every month or two. Each snapshot is named `CC-MAIN-<year>-<week>` and has about 3 billion web pages. Every snapshot comes in three formats -

1. WARC (Web ARChive). The raw HTTP responses, including headers and the full HTML of every page. WARC files also store the HTTP requests and some crawl metadata.
2. WAT. Metadata computed from the WARC records, such as HTTP headers and the links on each page, stored as JSON.
3. WET. Plain text extracted from the HTML of every page.

The files are stored in the `commoncrawl` bucket on Amazon S3 (us-east-1). A snapshot has tens of thousands of WARC files of about 1 GB each (gzip compressed), and one WAT and one WET file for every WARC file. Since April 2022, the bucket does not allow anonymous S3 access. We either use the S3 API with AWS credentials, or download the files over HTTPS from `https://data.commoncrawl.org/`. We shall use HTTPS.

We do not want to download hundreds of terabytes to find a few thousand recipe pages. Common Crawl provides two indexes for this -

1. The CDX index server at [index.commoncrawl.org](https://index.commoncrawl.org/). It answers queries like "all captures of URLs that match `www.food.com/recipe/*` in snapshot CC-MAIN-2023-14".
2. The columnar index. The same information as Parquet files, which we can query with SQL using Amazon Athena, Spark or DuckDB.

For every page, both indexes tell us the WARC file that contains it, the byte offset of the record inside that file and the length of the record. With these three numbers, we can download a single page with an HTTP Range request instead of the full 1 GB file.

## Our Use Case

We want a dataset of food recipes. [Food.com](https://www.food.com) has hundreds of thousands of user submitted recipes, and every recipe page has the same layout and a URL of the form `https://www.food.com/recipe/<name>-<id>`. We shall collect recipe pages from six snapshots: CC-MAIN-2023-06, CC-MAIN-2023-14, CC-MAIN-2023-23, CC-MAIN-2023-40, CC-MAIN-2023-50 and CC-MAIN-2024-10.

<div class="callout">
🤔 <b>Why not use a ready made recipe dataset?</b><br/>
There are recipe datasets on Kaggle and Hugging Face, some of them scraped from food.com. But the goal of this series is to learn how LLMs are built. The steps in this post (search the index, fetch the records, extract, filter, deduplicate) are the same steps used to build web datasets with trillions of tokens. Only the URL pattern and the scale change.
</div>

## Searching the Index

The CDX server takes a URL pattern and returns one JSON line per capture. Results are split into pages and we first ask for the number of pages. The server is shared by everyone and is often busy. It returns 503 or 504 errors, and sometimes it cuts a response short. So we ask for one index block per page (`pageSize=1`), retry with exponential back off, and fetch a page again if its last line is not valid JSON.

```python
import json
import time
import requests

CC_INDEX_SERVER = "https://index.commoncrawl.org"


def get_with_retries(url, params, max_retries=8):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                return None  # no captures for this page
        except requests.exceptions.RequestException:
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"Giving up on {url} {params}")


def search_cc_index(index_name, url_pattern="www.food.com/recipe/*"):
    endpoint = f"{CC_INDEX_SERVER}/{index_name}-index"
    params = {"url": url_pattern, "output": "json", "pageSize": 1}
    num_pages = get_with_retries(endpoint, {**params, "showNumPages": "true"}).json()["pages"]
    records = []
    for page in range(num_pages):
        for attempt in range(8):
            response = get_with_retries(endpoint, {**params, "page": page})
            if response is None:
                break
            try:
                records.extend([json.loads(line) for line in response.text.splitlines()])
                break
            except json.JSONDecodeError:
                time.sleep(2**attempt)  # truncated response, fetch the page again
    print(f"Found total {len(records)} in snapshot {index_name}")
    return records


for snapshot in ["CC-MAIN-2023-14", "CC-MAIN-2023-23", "CC-MAIN-2023-40", "CC-MAIN-2023-50", "CC-MAIN-2024-10"]:
    records = search_cc_index(snapshot)
# Found total 18969 in snapshot CC-MAIN-2023-14
# Found total 18342 in snapshot CC-MAIN-2023-23
# Found total 18479 in snapshot CC-MAIN-2023-40
# Found total 4469 in snapshot CC-MAIN-2023-50
# Found total 4630 in snapshot CC-MAIN-2024-10
```

Each record looks like this -

```python
print(records[0])
# {"urlkey": "com,food)/recipe/0-carb-0-cal-gummy-worms-283618", "timestamp": "20230322035457",
#  "url": "https://www.food.com/recipe/0-carb-0-cal-gummy-worms-283618", "mime": "text/html",
#  "mime-detected": "text/html", "status": "200", "digest": "2SYOGX2OSIYEPR6PVGV53Z4DONIEMR7D",
#  "length": "60630", "offset": "861832833",
#  "filename": "crawl-data/CC-MAIN-2023-14/segments/1679296943749.68/warc/CC-MAIN-20230322020215-20230322050215-00755.warc.gz",
#  "languages": "eng", "encoding": "UTF-8"}
```

(This record is from CC-MAIN-2023-14.) The `digest` is a hash of the page content, and `filename`, `offset` and `length` locate the record.

You will notice that CC-MAIN-2023-06 is missing from the list. For that snapshot, the server answered the page count query, but every request for the records failed with 504 Gateway Timeout, over several attempts. So for this snapshot we use the columnar index instead. It is a set of Parquet files, one set per snapshot. The rows are sorted by `url_surtkey` (the URL with the host name reversed, `com,food)/recipe/...`), and every Parquet file stores the minimum and maximum value of each column for every row group in its footer. DuckDB can read these footers over HTTPS and skip every file and row group that cannot contain food.com.

```python
import duckdb
import gzip
import requests

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

paths = gzip.decompress(
    requests.get("https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-06/cc-index-table.paths.gz").content
).decode().split()
paths = ["https://data.commoncrawl.org/" + p for p in paths if "subset=warc" in p]
print(len(paths))
# 300

# Read only the Parquet footers to find files that can contain food.com recipe pages
lo, hi = "com,food)/recipe/", "com,food)/recipe0"
hits = []
for url in paths:
    smallest, largest = con.execute(f"""
        SELECT min(stats_min_value), max(stats_max_value) FROM parquet_metadata('{url}')
        WHERE path_in_schema = 'url_surtkey'
    """).fetchone()
    if smallest <= hi and largest >= lo:
        hits.append(url)
print(len(hits))
# 1

df = con.execute(f"""
    SELECT url_surtkey AS urlkey, url, content_mime_type AS mime, fetch_status AS status,
           content_digest AS digest, warc_record_length AS length,
           warc_record_offset AS offset, warc_filename AS filename
    FROM read_parquet('{hits[0]}')
    WHERE url_surtkey >= '{lo}' AND url_surtkey < '{hi}'
""").df()
print(len(df))
# 4785
```

Only one of the 300 files can contain our URLs, and the final query took about 8 seconds. The `subset=warc` part of the columnar index holds only successful captures. Redirects and errors are in a separate `crawldiagnostics` subset. So all 4,785 rows have status 200.

The CDX results do include redirects (301, 308) and errors (404, 500, 504), and a few records whose MIME type is `unk`. We keep only records with status 200 and MIME type `text/html` -

| Snapshot | Index records | Status 200 and text/html |
| --- | --- | --- |
| CC-MAIN-2023-06 | 4,785 | 4,785 |
| CC-MAIN-2023-14 | 18,969 | 18,918 |
| CC-MAIN-2023-23 | 18,342 | 18,308 |
| CC-MAIN-2023-40 | 18,479 | 18,466 |
| CC-MAIN-2023-50 | 4,469 | 4,463 |
| CC-MAIN-2024-10 | 4,630 | 4,624 |
| Total | 69,674 | 69,564 |

The number of recipe pages differs a lot between snapshots. Common Crawl does not crawl every site to the same depth in every snapshot.

## Fetching WARC Records

To fetch one record, we send an HTTP GET with a `Range` header that covers `offset` to `offset + length - 1`. The server answers with status 206 (Partial Content) and the bytes of a single gzip member, which is a complete WARC record. [warcio](https://github.com/webrecorder/warcio) parses it.

```python
import io
from warcio.archiveiterator import ArchiveIterator

record = records[0]
start = int(record["offset"])
end = start + int(record["length"]) - 1
response = requests.get(
    "https://data.commoncrawl.org/" + record["filename"],
    headers={"Range": f"bytes={start}-{end}"},
)
print(response.status_code, len(response.content))
# 206 60630

for warc_record in ArchiveIterator(io.BytesIO(response.content)):
    print(warc_record.rec_type, warc_record.rec_headers.get_header("WARC-Target-URI"))
    html = warc_record.content_stream().read().decode("utf-8", errors="replace")
print(len(html))
# response https://www.food.com/recipe/0-carb-0-cal-gummy-worms-283618
# 381865
```

A single recipe page is 381,865 characters of HTML. The 60 KB we downloaded is the gzip compressed record.

Now we need to do the same for all 69,564 records. Every request downloads only about 60 KB and most of the time is spent waiting for the server. So we send many requests at the same time. We use [Dask](https://www.dask.org/) for this. Dask starts a local cluster of worker processes, each with a few threads, and runs our Python functions on them. A Dask bag is a list of Python objects split into partitions that are processed in parallel. We split the records of a snapshot into chunks of 40 and every chunk becomes one task. Every task writes its results to its own file. If the run stops halfway, the next run skips the chunks that are already done.

```python
import os
import random
import threading
import time

import dask.bag as db
from dask.distributed import Client, LocalCluster

thread_local = threading.local()
MAX_REQUESTS_PER_SECOND = 1.75  # per worker process
rate_lock = threading.Lock()
next_request_time = [0.0]


def wait_for_turn():
    # Allow at most MAX_REQUESTS_PER_SECOND requests per second from all threads of this process
    with rate_lock:
        now = time.time()
        wait = next_request_time[0] - now
        next_request_time[0] = max(now, next_request_time[0]) + 1 / MAX_REQUESTS_PER_SECOND
    if wait > 0:
        time.sleep(wait)


def fetch_warc_record(record, max_retries=10):
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    start = int(record["offset"])
    end = start + int(record["length"]) - 1
    for attempt in range(max_retries):
        wait_for_turn()
        try:
            response = thread_local.session.get(
                "https://data.commoncrawl.org/" + record["filename"],
                headers={"Range": f"bytes={start}-{end}"},
                timeout=60,
            )
            if response.status_code == 206:
                return response.content
            # 503 (SlowDown) and 403 (request blocked by the CDN) mean we are too fast
        except requests.exceptions.RequestException:
            pass
        time.sleep(min(2**attempt, 120) + random.random())
    return None


def process(record):
    result = {"url": record["url"], "snapshot": record["snapshot"], "digest": record["digest"]}
    warc_bytes = fetch_warc_record(record)
    if warc_bytes is None:
        result["error"] = "fetch_failed"
        return result
    for warc_record in ArchiveIterator(io.BytesIO(warc_bytes)):
        if warc_record.rec_type == "response":
            page = warc_record.content_stream().read().decode("utf-8", errors="replace")
            recipe = extract_recipe(page)  # defined in the next section
            if recipe is None:
                result["error"] = "no_recipe_json_ld"
            else:
                result.update(recipe)
            return result
    result["error"] = "no_response_record"
    return result


def process_chunk(chunk):
    out_path, records = chunk
    if os.path.exists(out_path):
        return 0  # done in an earlier run
    lines = [json.dumps(process(record)) for record in records]
    with open(out_path + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(out_path + ".tmp", out_path)
    return len(lines)


if __name__ == "__main__":
    client = Client(LocalCluster(n_workers=4, threads_per_worker=8))
    for snapshot in SNAPSHOTS:
        records = load_index(snapshot)  # index records with status 200 and mime text/html
        out_dir = os.path.join("raw", snapshot)
        os.makedirs(out_dir, exist_ok=True)
        chunks = [
            (os.path.join(out_dir, f"part-{i // 40:05d}.jsonl"), records[i:i + 40])
            for i in range(0, len(records), 40)
        ]
        db.from_sequence(chunks, npartitions=len(chunks)).map(process_chunk).sum().compute()
```

The rate limit was not in my first version, and that was a mistake. data.commoncrawl.org is served through a CDN (Amazon CloudFront) in front of S3. With 64 threads (about 33 requests per second), every request soon failed with status 403 and a short HTML page that said "Request blocked". For a few minutes, even small files like `warc.paths.gz` were blocked. A limit of 12 requests per second was blocked after about 3 minutes. 7 requests per second in total (4 workers, 1.75 requests per second each) was not blocked. A 403 usually means "you are not allowed", but here it means "slow down", so we retry it with back off.

There is a second kind of error that does not depend on us. When many people download from Common Crawl at the same time, S3 answers with status 503 and the error code `SlowDown`. During our run there was a period of about half an hour when most requests got a 503. The back off handles these too, but some records failed all 10 attempts. After the main run, we fetch the failed records again with the same function.

In the main run, 142 records failed all 10 attempts (34 in CC-MAIN-2023-06, 55 in CC-MAIN-2023-23, 17 in CC-MAIN-2023-50 and 36 in CC-MAIN-2024-10). All of them worked in the second pass, so we have all 69,564 pages. At about 6 records per second, the 18,466 records of CC-MAIN-2023-40 took 53 minutes. The six snapshots took a little over 3 hours in total.

## Extracting the Recipe

Now we need the recipe text from 381,865 characters of HTML. There are three options -

1. Use the WET files that Common Crawl already extracted.
2. Extract the main content of the HTML with a library like trafilatura, as RefinedWeb does.
3. Use structured data in the page.

Let's look at the WET text first. The CDX index does not point into WET files, but every WARC file has a WET file with the same name in the same segment. So I streamed the 121 MB WET file for the WARC file above until I found our page (record 28,889 in the file). The text has 187 lines and 4,978 characters. Here is a part of it -

```
0 Carb &amp; 0 Cal Gummy Worms!! Recipe - Food.com
Recipes
Breakfast & Brunch Recipes
Lunch Recipes
... (about 80 more lines of the site menu)
icons / ellipsis / ellipsis-horizontal
save
Download
Print
Share
I Made This
photo by Feltz S.
Ready In:
45mins
Ingredients:
3
Serves:
6
Nutrition information
Advertisement
ingredients
Units: US
2 (6 ounce) packages sugar-free jello
2 (6 ounce) packages plain gelatin
1 cup boiling water
Advertisement
directions
Stir all ingredients until dissolved.
...
Questions & Replies
Sign In
to Ask a Question
Got a question? Share it with the community!
Advertisement
Reviews
MOST POPULARMOST RECENT
Write A Review
...
© 2023 Warner Bros. Discovery, Inc. or its subsidiaries and affiliates. All rights reserved.
Advertise
AdChoices
Privacy Notice
Visitor Agreement
California Privacy Notice
Do Not Sell or Share My Personal Information
```

The recipe itself is about a dozen lines. The rest is the site menu, buttons, `Advertisement` markers, reviews, the profile of the recipe author and the page footer. Every food.com page repeats the same menu and footer. A model trained on this text would spend a lot of its capacity learning to write `icons / ellipsis / ellipsis-horizontal`.

trafilatura removes most of the menu and the footer -

```python
import trafilatura

text = trafilatura.extract(html)
print(len(text))
print("\n".join(line.strip() for line in text.splitlines() if line.strip()))
# 1675
# 0 Carb & 0 Cal Gummy Worms!!
# photo by Feltz S.
# - Ready In:
# - 45mins
# - Ingredients:
# - 3
# - Serves:
# -
# 6
# ingredients
# - 2 (6 ounce) packages sugar-free jello
# - 2 (6 ounce) packages plain gelatin
# - 1 cup boiling water
# directions
# - Stir all ingredients until dissolved.
# ...
# Questions & Replies
# Got a question?
# Share it with the community!
# Reviews
# RECIPE SUBMITTED BY
# ... (the profile of the recipe author follows)
```

This is much better, but it still keeps parts of the page around the recipe. To check that this is not a problem of one page, I ran trafilatura on a random sample of 200 records from CC-MAIN-2023-50 (192 of them downloaded). 185 of the 192 outputs still had the "Questions & Replies" block and 156 had the "RECIPE SUBMITTED BY" block with the profile of the recipe author. trafilatura also took 172 ms per page on my laptop. That is about 3.3 CPU hours for our 69,564 pages.

The third option is the best for this site. Like many recipe sites, food.com embeds the recipe in the page as [schema.org Recipe](https://schema.org/Recipe) data inside a `<script type="application/ld+json">` tag, so that search engines can show recipe cards. It looks like this (shortened) -

```
{"@context": "http://schema.org", "@type": "Recipe",
 "name": "0 Carb &amp; 0 Cal Gummy Worms!!",
 "description": "these are delicious and guilt free! ...",
 "recipeIngredient": ["2 (6   ounce) packages   sugar-free jello", "2 (6   ounce) packages  plain gelatin", "1   cup    boiling water"],
 "recipeInstructions": [{"@type": "HowToStep", "text": "Stir all ingredients until dissolved. \r"}, ...],
 "aggregateRating": {...}, "nutrition": {...}, "review": [...], ...}
```

The title, the ingredients and the steps are already separated for us. Other sites use slightly different shapes of the same schema. The Recipe object can be inside a list or an `@graph`, and instructions can be a string, a list of strings, `HowToStep` objects or `HowToSection` objects that group steps. The following code handles all of these.

```python
import html as html_lib
import json
import re

JSON_LD_PATTERN = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S)


def find_recipe(node):
    if isinstance(node, list):
        for item in node:
            found = find_recipe(item)
            if found:
                return found
    elif isinstance(node, dict):
        node_type = node.get("@type")
        if node_type == "Recipe" or (isinstance(node_type, list) and "Recipe" in node_type):
            return node
        if "@graph" in node:
            return find_recipe(node["@graph"])
    return None


def flatten_instructions(instructions):
    if isinstance(instructions, str):
        return [instructions]
    steps = []
    for item in instructions or []:
        if isinstance(item, str):
            steps.append(item)
        elif isinstance(item, dict):
            if "itemListElement" in item:  # HowToSection
                steps.extend(flatten_instructions(item["itemListElement"]))
            elif "text" in item:  # HowToStep
                steps.append(item["text"])
    return steps


def extract_recipe(page):
    for match in JSON_LD_PATTERN.finditer(page):
        try:
            recipe = find_recipe(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
        if recipe:
            return {
                "title": html_lib.unescape(recipe.get("name") or ""),
                "ingredients": [html_lib.unescape(i) for i in recipe.get("recipeIngredient") or []],
                "directions": [html_lib.unescape(s) for s in flatten_instructions(recipe.get("recipeInstructions"))],
            }
    return None
```

Note the `html.unescape`. The JSON-LD text still has HTML entities like `&amp;`.

On the same 192 pages, `extract_recipe` took 0.4 ms per page, more than 400 times faster than trafilatura. Over all 69,564 pages, only 3 pages had no Recipe object, and none of them is a recipe. One is `https://www.food.com/recipe/all/healthy`, a list of healthy recipes. The other two have the URLs `https://www.food.com/recipe/all/%7b%7b=it.itemurl%7d%7d` and `.../%7b%7b=it.userprofileurl%7d%7d`. `%7b` is an encoded `{` and `%7d` is `}`, so these URLs are template code like `it.itemurl` in double curly brackets. The crawler most likely found these links inside a JavaScript template and followed them as they were.

This works only because we collect pages from one site. A general web dataset cannot depend on structured data and must use a tool like trafilatura, with all its noise.

## Formatting the Text

The model will see plain text, so we need one fixed format for every recipe. We use the same format that the model generates in post 4. It has a title, a blank line, the ingredients one per line, a blank line and the directions one per line.

```python
import unicodedata


def normalise_line(line):
    line = unicodedata.normalize("NFC", line)
    # Some recipes link to other recipes with <a href="...">name</a>. Keep only the name.
    line = re.sub(r"<[^>]+>", " ", line)
    return " ".join(line.split())


def format_recipe(doc):
    doc["title"] = normalise_line(doc["title"])
    doc["ingredients"] = [i for i in map(normalise_line, doc["ingredients"]) if i]
    doc["directions"] = [s for s in map(normalise_line, doc["directions"]) if s]
    doc["text"] = (
        f"Title: {doc['title']}\n\n"
        "Ingredients:\n" + "\n".join(doc["ingredients"]) + "\n\n"
        "Directions:\n" + "\n".join(doc["directions"])
    )
    return doc


print(format_recipe(extract_recipe(html))["text"])
# Title: 0 Carb & 0 Cal Gummy Worms!!
#
# Ingredients:
# 2 (6 ounce) packages sugar-free jello
# 2 (6 ounce) packages plain gelatin
# 1 cup boiling water
#
# Directions:
# Stir all ingredients until dissolved.
# Pour the mixture onto a large dinner plate and refrigerate.
# It will set in about 20 minutes.
# You can either slice it into worms, or roll up the rubbery disk of gelatin and cut it every 1/4 inch with a large pair of scissors.
# You can also use tiny cutters to make little shapes.
# Total Recipe: 50 Cal (0% from Fat, 100% from Protein, 0% from Carb); 12 g Protein; 0 g Tot Fat; 0 g Carb; 0 g Fiber; 5 mg.
# Calcium; 0 mg Iron; 7 mg Sodium; 0 mg Cholesterol.
```

The 381,865 characters of HTML became 645 characters of text. `normalise_line` also collapses the runs of spaces in strings like `"2 (6   ounce) packages   sugar-free jello"` and removes the `\r` at the end of every step. We leave out the description, the reviews and the nutrition facts. They are useful text, but the model should learn to write recipes, not reviews.

## Filtering

Even clean recipe text needs some filtering. Let's start with the rules of C4, as listed in the T5 paper -

1. Keep only lines that end with a terminal punctuation mark (a full stop, exclamation mark, question mark or end quotation mark).
2. Remove pages with fewer than 3 sentences, and keep only lines with at least 5 words.
3. Remove pages that contain any word from the "List of Dirty, Naughty, Obscene or Otherwise Bad Words".
4. Remove lines that contain the word "Javascript".
5. Remove pages that contain the phrase "lorem ipsum".
6. Remove pages that contain a curly bracket `{`, since it appears in most programming languages.
7. Remove citation markers like `[1]` and `[citation needed]`.
8. Remove lines that contain "terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies" or "use cookies".
9. Keep pages that langdetect classifies as English with a probability of at least 0.99.
10. Remove all but one copy of any three sentence span that occurs more than once in the dataset.

Rules 1 to 8 are simple text rules. Here they are in code (rule 7 is for Wikipedia pages, so we skip it) -

```python
BAD_WORDS = [w for w in open("badwords_en.txt").read().split("\n") if w]
# A bad word or phrase with a non-word character (or the start or end of the text) on each side
BAD_WORDS_PATTERN = re.compile(
    r"(?:\W|^)(" + "|".join(re.escape(w) for w in sorted(BAD_WORDS, key=len, reverse=True)) + r")(?:\W|$)"
)
POLICY_STRINGS = ["terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies"]
TERMINAL_PUNCTUATION = (".", "!", "?", '"')


def c4_clean_page(text):
    if "lorem ipsum" in text.lower() or "{" in text:
        return None
    if BAD_WORDS_PATTERN.search(text.lower()):
        return None
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.endswith(TERMINAL_PUNCTUATION):
            continue
        if len(line.split()) < 5:
            continue
        if "javascript" in line.lower():
            continue
        if any(p in line.lower() for p in POLICY_STRINGS):
            continue
        lines.append(line)
    if len(re.findall(r"[.!?]+(\s|$)", " ".join(lines))) < 3:
        return None
    return "\n".join(lines)
```

`badwords_en.txt` is the [English list](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) used by C4. It has 403 entries and 124 of them are phrases, so we match it with a regular expression like the C4 code does.

Let's apply `c4_clean_page` to all our recipes as it is. The fetch step saved one JSON line per page in `raw/<snapshot>/part-XXXXX.jsonl`, and Dask can read all of them as one bag -

```python
docs = (
    db.read_text("raw/*/*.jsonl")
    .map(json.loads)
    .filter(lambda d: "error" not in d)  # the 3 pages without a Recipe object
    .map(format_recipe)
    .compute()
)
print(len(docs))
# 69561

cleaned = [c4_clean_page(doc["text"]) for doc in docs]
print(sum(c is not None for c in cleaned))
print(round(sum(len(c) for c in cleaned if c) / sum(len(doc["text"]) for doc in docs), 3))
# 65301
# 0.618
```

C4 keeps 65,301 pages (94%) but only 61.8% of the characters. Let's see where the rest goes -

1. Ingredient lines. A line like `2 (6 ounce) packages sugar-free jello` has no punctuation mark at the end. Only 38 of 661,615 ingredient lines pass the line rules. So almost every recipe that C4 keeps has lost all of its ingredients. Direction lines do better, 423,570 of 475,617 pass.
2. Short recipes. 3,961 pages have fewer than 3 sentences after the line rules. Many short recipes are fine, for example a drink with "Blend and serve in a hurricane glass." as its only step.
3. Bad words. 260 pages contain an entry from the list. All of them are recipes, and most of the matches are normal food words. The most common word is "butt" (142 pages, from pork butt). Next are "sex" (20 pages, Sex on the Beach cocktails and Better Than Sex cakes), "hard core" (11 pages, the hard core of a cabbage), "twinkie" (11), "rimming" (8, rimming a margarita glass with salt), "dick" (4, the British pudding spotted dick) and "cock" (3, Cock-A-Leekie Soup).
4. Curly brackets. 39 pages. None of them has code. Some recipe authors use curly brackets like round brackets, for example `fry eggs as you like {scrambled, fried, etc.}`.
5. Policy lines. 9 lines contain "use cookies", but not one is a cookie notice. 8 of them are titles like "Better Than Toll House Cookies!", where "house cookies" contains "use cookies". The last one is "I like to use cookies sheets."
6. Language. With the 0.99 threshold, langdetect rejects 103 pages. In 100 of them, English is still the most likely language. The probabilities are 0.857, 0.714 and 0.571, which are 6/7, 5/7 and 4/7. langdetect runs 7 random trials by default and reports the fraction of trials for each language. On short texts with many numbers and units, the trials do not always agree.

C4 also removes repeated three sentence spans (rule 10). We do not need this, since we remove duplicate recipes in the next section.

The C4 rules were written for text extracted from any web page. In that text, a line without a punctuation mark at the end is usually a menu item or a button. Our text has no menus, so the line rules only remove ingredients. The bad word list removes food. [Dodge et al. (2021)](https://arxiv.org/abs/2104.08758) found a worse side effect in C4 itself. The list removed text written in African American English and text about LGBTQ+ people at a much higher rate than other text. A word list is a weak way to find harmful text.

So we shall not use the line rules or the word list. Instead, we write a few page level rules for our data. Every rule returns a reason, so that we can count what each rule removes and look at the pages it removes -

```python
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # langdetect is random, make it repeatable
POLICY_PATTERN = re.compile(r"\b(" + "|".join(POLICY_STRINGS + ["javascript"]) + r")\b")


def filter_reason(doc):
    text = doc["text"]
    if not doc["ingredients"] or not doc["directions"]:
        return "missing_ingredients_or_directions"
    if "lorem ipsum" in text.lower():
        return "lorem_ipsum"
    if "{" in text:
        return "curly_bracket"
    if POLICY_PATTERN.search(text.lower()):
        return "boilerplate"
    if re.search(r"https?://|www\.", text):
        return "contains_url"
    try:
        if detect_langs(text)[0].lang != "en":
            return "not_english"
    except Exception:  # langdetect fails on text without letters
        return "not_english"
    return None


docs = (
    db.read_text("raw/*/*.jsonl")
    .map(json.loads)
    .filter(lambda d: "error" not in d)
    .map(format_recipe)
    .map(lambda d: {**d, "filter": filter_reason(d)})
    .persist()
)
print(docs.map(lambda d: d["filter"]).frequencies().compute())
# [(None, 69353), ('contains_url', 165), ('curly_bracket', 39), ('not_english', 3), ('boilerplate', 1)]
```

`persist()` keeps the results in the memory of the Dask workers, so we can run more than one query on them without computing everything again. The rules removed 208 pages -

1. `contains_url` removed 165 pages. Most of them end with a link, like `Read more: http://www.food.com/recipe/...`, a YouTube video or a shop that sells an ingredient. The model cannot follow a link, and we do not want it to write them.
2. `curly_bracket` removed 39 pages. As we saw, these are real recipes. We remove them anyway because the loss is small.
3. `not_english` removed 3 pages. One is a Chilean salad with all the steps in both English and Spanish. The other two are short English drink recipes, "Tequila Bay Breeze" and "Pina Colada Lemonade". Language identification is not reliable on very short text.
4. `boilerplate` removed 1 page, the one that says "I like to use cookies sheets." The word boundaries `\b` stop "Toll House Cookies" from matching, but this line has the real words "use cookies".
5. `missing_ingredients_or_directions` and `lorem_ipsum` removed nothing.

So 3 of the 208 removed pages are mistakes, and 39 more are recipes we chose to drop. Every rule based filter removes some good data. What matters is to look at what it removes before we trust it.

## Deduplication

Duplicates hurt in two ways. They waste compute, and the model memorises text it sees many times instead of learning from it. [Lee et al. (2021)](https://arxiv.org/abs/2107.06499) found that removing duplicates from C4 reduced the rate at which models output memorised training text by about 10 times, and did not hurt the perplexity.

I expected the six snapshots to overlap a lot, since a popular recipe page should be crawled again and again. Let's count the recipe URLs that appear in both snapshots, for every pair of snapshots -

```python
SNAPSHOTS = ["CC-MAIN-2023-06", "CC-MAIN-2023-14", "CC-MAIN-2023-23",
             "CC-MAIN-2023-40", "CC-MAIN-2023-50", "CC-MAIN-2024-10"]
docs = docs.filter(lambda d: d["filter"] is None).compute()
urls = {s: {d["url"] for d in docs if d["snapshot"] == s} for s in SNAPSHOTS}
for a in SNAPSHOTS:
    print(a, [len(urls[a] & urls[b]) for b in SNAPSHOTS])
# CC-MAIN-2023-06 [4766, 9, 15, 18, 9, 25]
# CC-MAIN-2023-14 [9, 18836, 16, 19, 8, 15]
# CC-MAIN-2023-23 [15, 16, 18237, 15, 19, 18]
# CC-MAIN-2023-40 [18, 19, 15, 18408, 17, 12]
# CC-MAIN-2023-50 [9, 8, 19, 17, 4437, 5]
# CC-MAIN-2024-10 [25, 15, 18, 12, 5, 4610]
```

The numbers on the diagonal are the unique recipe URLs in each snapshot. The rest are tiny. Out of 18,408 recipes in CC-MAIN-2023-40, only 19 are also in CC-MAIN-2023-14. I was wrong. Common Crawl does not crawl the same pages again and again. Its crawler seems to pick a different sample of URLs for every crawl, and every crawl announcement reports how many URLs were not in any earlier crawl. For us this is good news, since six snapshots give us six mostly different sets of recipes.

![Recipes per snapshot](/assets/images/Data_Collection_and_Preparation/1.png)

In total, 219 pages are a recipe we already have from an earlier snapshot, and 59 more are the same URL captured twice in the same snapshot.

### Exact Duplicates

We remove exact duplicates in two steps. First by URL, keeping the capture from the latest snapshot. Then by a hash of the text, to catch the same recipe under two different URLs.

```python
import hashlib

# 1. Same URL. Keep the capture from the latest snapshot.
docs.sort(key=lambda d: SNAPSHOTS.index(d["snapshot"]), reverse=True)
by_url = {}
for d in docs:
    by_url.setdefault(d["url"], d)
docs = list(by_url.values())
print(len(docs))
# 69075

# 2. Same text
by_hash = {}
for d in docs:
    by_hash.setdefault(hashlib.sha1(d["text"].encode()).hexdigest(), d)
docs = list(by_hash.values())
print(len(docs))
# 69004
```

When a URL had more than one capture, the recipe text was the same in every capture. But the `digest` values in the index were all different. The digest is a hash of the full HTML, and the ads, reviews and scripts around the recipe change between captures. So a hash of the raw page is not a good key to find duplicates. A hash of the extracted text is.

The text hash removed 71 more pages. In 61 of them, the URL had a query string, like `lemon-bars-9989?photo=35852` and `lemon-bars-9989`, or `white-pizza-sauce-279060?ftab=reviews`. Removing the query string from the URLs before the first step would also catch these. In the other 10, the same recipe was posted twice with two recipe ids, often consecutive ones like `grapes-in-port-wine-sauce-534699` and `grapes-in-port-wine-sauce-534700`.

### Near Duplicates

Some recipes are posted twice with small changes. For example, "Szechwan Shrimp" (recipe 287913) and "Sweet & Salty Sezchuan Shrimp" (recipe 188344) differ only in the title. A hash cannot find these.

We need a similarity measure for two texts. A common one is the Jaccard similarity of their sets of shingles. A shingle is a sequence of $n$ consecutive words. With $n = 5$, "Stir all ingredients until dissolved." gives the shingle "stir all ingredients until dissolved." and longer texts give one shingle for every position. For the shingle sets $A$ and $B$ of two documents -

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Computing $J$ for every pair of our 69,004 recipes means 2.4 billion comparisons of sets. MinHash and locality sensitive hashing (LSH) avoid this. Andrei Broder developed MinHash in 1997 to find near duplicate pages in the web crawl of the AltaVista search engine. Together with LSH, it was used to deduplicate the training data of GPT-3, The Pile, RefinedWeb and SlimPajama.

MinHash works like this. Take a random hash function $h$, hash every shingle of a document and keep only the smallest value. For two documents $A$ and $B$, the smallest value is the same with probability exactly $J(A, B)$, because the shingle with the smallest hash in $A \cup B$ is equally likely to be any of them, and the two minimums are equal only when it is in $A \cap B$. With 128 different hash functions, every document gets a signature of 128 numbers, and the fraction of positions where two signatures agree is an estimate of $J$.

LSH finds the pairs with similar signatures without comparing all pairs. It splits the signature into $b$ bands of $r$ numbers each. Two documents become a candidate pair if all $r$ numbers are equal in at least one band. Documents are put in a hash table per band, so this costs about the same as reading the signatures once. If the similarity of two documents is $s$, the probability that they become a candidate pair is

$$P(s) = 1 - (1 - s^r)^b$$

For a threshold of 0.8 and 128 hash functions, datasketch chooses $b = 9$ and $r = 13$ -

| Jaccard similarity $s$ | 0.5 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| $P(s)$ | 0.001 | 0.084 | 0.399 | 0.686 | 0.929 | 0.998 |

Pairs with $s \geq 0.9$ are almost always found and pairs with $s \leq 0.7$ are rarely found. Pairs at exactly 0.8 are found only 40% of the time, so this setting will miss some near duplicates. More bands would find more of them, but would also give more false candidates to check.

Computing 128 hashes for every shingle is the slow part, so we do it in parallel with Dask. Building the LSH index is fast and runs in a single process -

```python
from datasketch import LeanMinHash, MinHash, MinHashLSH


def word_shingles(text, n=5):
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def compute_minhash(doc):
    m = MinHash(num_perm=128, seed=1)
    m.update_batch([s.encode("utf-8") for s in word_shingles(doc["text"])])
    return doc["id"], LeanMinHash(m)


for i, d in enumerate(docs):
    d["id"] = i
signatures = db.from_sequence(docs, npartitions=64).map(compute_minhash).compute()

lsh = MinHashLSH(threshold=0.8, num_perm=128)
print(lsh.b, lsh.r)
# 9 13

minhashes, keep, near_pairs, rejected = {}, [], [], 0
for doc_id, m in signatures:
    # LSH gives candidates. Keep only those with an estimated Jaccard similarity of at least 0.8.
    scored = [(m.jaccard(minhashes[c]), c) for c in lsh.query(m)]
    rejected += sum(1 for j, c in scored if j < 0.8)
    scored = [(j, c) for j, c in scored if j >= 0.8]
    if scored:
        near_pairs.append((doc_id, max(scored)))
        continue
    lsh.insert(doc_id, m)
    minhashes[doc_id] = m
    keep.append(doc_id)
print(len(keep), len(near_pairs), rejected)
# 68978 26 6
```

`LeanMinHash` is a smaller copy of the signature without the hash functions, so it is cheaper to send from the Dask workers back to the main process.

MinHash removed 26 near duplicates. The 6 rejected candidates had an estimated similarity between 0.67 and 0.80. They are variants of a recipe rather than copies. "Pecan Kringle" and "Double Berry Kringle" (0.71) have the same pastry and a different filling. "Hunza Bread" and "Hunza Diet Bread Recipe" (0.77) have the same steps, but one uses half of every ingredient. Whether to remove such pairs is a choice, and we keep them. Here is one of the near duplicates we removed (0.91) -

```
--- no-knead-bread-348028
+++ new-york-times-no-knead-bread-464732
-Title: No-Knead Bread
+Title: New York Times No-Knead Bread
-3 cups all-purpose flour, more for dusting
+3 cups all-purpose flour or 3 cups bread flour, more for dusting
-cornmeal, as needed
+cornmeal or wheat bran, as needed
```

All other lines of the two recipes are the same.

Exact and near deduplication together removed only 375 of 69,353 pages (0.5%). This is much less than SlimPajama, which removed almost half of RedPajama. Their data mixed many sources and snapshots that overlap a lot. We have one site where every recipe has its own URL, and snapshots that barely overlap.

## Writing the Dataset

The next post reads JSON lines files with a `text` field from `~/EduLLM/data/food-com-cc-cleaned/`. We sort the recipes by URL and let Dask write them as 8 shards -

```python
import math
import numpy as np

final = [docs[i] for i in keep]
out_dir = os.path.expanduser("~/EduLLM/data/food-com-cc-cleaned")
os.makedirs(out_dir, exist_ok=True)
(
    db.from_sequence(sorted(final, key=lambda d: d["url"]), partition_size=math.ceil(len(final) / 8))
    .map(lambda d: json.dumps({"text": d["text"]}))
    .to_textfiles(os.path.join(out_dir, "*.jsonl"))
)
print(sorted(os.listdir(out_dir)))
# ['0.jsonl', '1.jsonl', '2.jsonl', '3.jsonl', '4.jsonl', '5.jsonl', '6.jsonl', '7.jsonl']

words = [len(d["text"].split()) for d in final]
size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
print(f"recipes {len(final)}, words {sum(words)}, characters {sum(len(d['text']) for d in final)}, size {size / 1e6:.1f} MB")
print(np.percentile(words, [5, 25, 50, 75, 95]))
# recipes 68978, words 11323175, characters 65737323, size 68.2 MB
# [ 58. 101. 143. 201. 338.]
```

Here is the whole pipeline with the number of pages after each step -

| Step | Pages |
| --- | --- |
| CDX and columnar index search | 69,674 |
| Status 200 and text/html | 69,564 |
| Fetched and has a Recipe object | 69,561 |
| Our filters | 69,353 |
| URL deduplication | 69,075 |
| Exact text deduplication | 69,004 |
| MinHash near deduplication | 68,978 |

The final dataset has 68,978 recipes, 11.3 million words and 68.2 MB of text. The next post splits the text on spaces only, which gives about 10.1 million words. An average recipe has 9.5 ingredient lines and 6.8 direction lines. Half of the recipes have between 101 and 201 words.

![Words per recipe](/assets/images/Data_Collection_and_Preparation/2.png)

For scale, C4 has about 750 GB of text, more than 10,000 times more. But our model and our goal are much smaller too. We want a model that writes recipes, not one that knows everything.

## Synthetic Data

Web data is not the only option. In 2023, two papers from Microsoft Research showed that small models can learn a lot from text written by a larger LLM.

[TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023) is a dataset of short stories that GPT-3.5 and GPT-4 wrote using only words a 3 to 4 year old child understands. To get variety, every prompt asked for a story that uses three random words (a noun, a verb and an adjective) and a few story features, such as a dialogue or a bad ending. Models with fewer than 10 million parameters trained on TinyStories write fluent and consistent stories, which models of that size trained on web text cannot do.

[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) (phi-1, 2023) trained a 1.3 billion parameter code model on about 7 billion tokens. Most of it was "textbook quality" code from the web, selected by a classifier trained on GPT-4 labels. The rest, under 1 billion tokens, was synthetic Python textbooks written by GPT-3.5. The model was then fine-tuned on a small set of synthetic exercises. phi-1 reached 50.6% pass@1 on HumanEval, better than many much larger models trained on much more data.

Both papers show that the quality and the focus of the data can matter as much as its size. Our setup is similar to TinyStories, with a narrow domain and a small model. We could ask an LLM to write more recipes in the same format. But synthetic data has its own problems. The variety is limited by the prompts, mistakes of the larger model are copied into the dataset, and the terms of use of the larger model may not allow it. For this series, we stay with real recipes.

## Next Steps

All code used in this post can be found on the associated [GitHub repository](https://github.com/gauravtendolkar/EduLLM).

In the next post, we shall train a byte pair encoding (BPE) tokeniser on the recipes in `~/EduLLM/data/food-com-cc-cleaned/`.
