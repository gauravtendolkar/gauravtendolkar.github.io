# gauravtendolkar.github.io

Personal site and blog, built with Jekyll from `docs/` and published by GitHub Pages.

## Writing and publishing (browser, no setup)

1. Go to <https://app.pagescms.org> and **Sign in with GitHub**.
2. Open `gauravtendolkar/gauravtendolkar.github.io` (only accounts with write access to this repo can open it).
3. **Blog posts → Add an entry**, fill in:
   - **Title**: number posts in a series, e.g. `7. My Post Title` (series lists are sorted by title).
   - **Date shown on the site**: free text, e.g. `June 3, 2025` or `Coming Soon...`.
   - **Series**: e.g. `Super-Fast-LLM-Training`, `Deep-Generative-Modelling`, or a new slug to start a new series.
   - **Live**: off = listed as an upcoming, unclickable post; on = published.
   - **Featured**: pins the post at the top of the home page (otherwise the newest live post is shown).
   - **Content**: Markdown (see below).
4. **Save**. That commits to `main`, and GitHub Pages republishes in about a minute.

Books and Apps are edited the same way (the **Books** and **Apps** entries in the sidebar).

### Content syntax

| What | How |
| --- | --- |
| Inline / display math | `$x^2$`, `$$\sum_i x_i$$` (MathJax) |
| Code | ```` ```python ```` fences, or `{% highlight python %} … {% endhighlight %}` |
| Diagrams | ```` ```mermaid ```` fences |
| Callout box | `<div class="callout"> … </div>` |
| Images | Upload in **Media**, then `![alt](/assets/images/file.png)` |

The first paragraph of a post is shown as the large intro next to the "Published" date.

### Post illustrations

Each post's illustration is `docs/assets/images/art/<post-file-slug>.svg` (for example `2-Training-Custom-BPE-Tokeniser.svg`).
To use your own image, upload a `.png`, `.jpg` or `.webp` with the same name, which takes priority, or set `image: /assets/images/…` in the post's front matter.
Posts without art fall back to `default.svg`.

## Local preview

With Docker Desktop running:

```powershell
./serve.ps1          # http://localhost:4000
docker rm -f gt-jekyll   # stop
```

## Layout

- `docs/_layouts`: `home` (blog index), `post`, `page`, `default`
- `docs/_includes`: `head`, `header`, `footer`, `art`
- `docs/assets/css/site.css`: the whole theme
- `docs/_data/books.yml`, `docs/_data/apps.yml`: Books and Apps content
- `.pages.yml`: Pages CMS (editor) configuration
