"""爬取 Dcard NCCU 版「找課/選課」相關文章 - v2 browser mode。

用 Playwright headless chromium，直接開瀏覽器頁面，
intercept network response 抓 API JSON，繞過 Cloudflare。

Output: data/raw/dcard_real_queries.jsonl
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright, Route, Request

ROOT = Path(__file__).resolve().parents[1]
FORUM = "nccu"

ARTICLE_KEYWORDS = [
    "找課", "推薦課", "哪門課", "哪堂課", "哪堂", "選課問題",
    "有沒有", "求推", "課程推薦", "哪個老師", "哪位老師",
    "想修", "想學", "好的課", "好課", "好玩的課", "輕鬆的課",
    "通識推薦", "體育課", "英文課", "求推薦", "推薦嗎",
    "哪門比較好", "哪堂比較好",
]

QUERY_PATTERNS = [
    r"有沒有[^，。！？\n]{5,40}[課程]",
    r"想[找修學][^，。！？\n]{4,35}",
    r"推薦[^，。！？\n]{4,35}[課程]",
    r"哪[門堂個位][^，。！？\n]{4,35}",
    r"[找求]好[的一個][^，。！？\n]{3,30}[課程]",
    r"[不要不想][^，。！？\n]{3,30}[課程]",
    r"[週一二三四五六日][^，。！？\n]{3,30}[課程]",
    r"[英中日韓]文授課[^，。！？\n]{3,25}",
    r"[幾\d][學分][^，。！？\n]{2,20}[課程]",
    r"不要早八[^，。！？\n]{0,30}",
    r"[下午晚上早上][^，。！？\n]{3,25}[課程]",
]


def is_relevant(title: str, excerpt: str = "") -> bool:
    text = title + excerpt
    return any(kw in text for kw in ARTICLE_KEYWORDS)


def extract_queries(text: str) -> list[str]:
    queries = []
    for pattern in QUERY_PATTERNS:
        for m in re.findall(pattern, text):
            m = m.strip()
            if 8 <= len(m) <= 50:
                queries.append(m)
    return list(dict.fromkeys(queries))


def crawl(n_pages: int, out_path: Path) -> list[dict]:
    results: list[dict] = []
    seen_queries: set[str] = set()
    qid_counter = 1
    captured_posts: list[dict] = []

    def handle_response(response):
        url = response.url
        if f"/forums/{FORUM}/posts" in url and "api/v2" in url:
            try:
                data = response.json()
                if isinstance(data, list):
                    captured_posts.extend(data)
            except Exception:
                pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="zh-TW",
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()
        page.on("response", handle_response)

        # 先開 Dcard NCCU 版
        print(f"[crawl] opening Dcard NCCU board...")
        try:
            page.goto(f"https://www.dcard.tw/f/{FORUM}", wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"[warn] goto timeout (continuing): {e}")
        time.sleep(5)

        print(f"[crawl] captured {len(captured_posts)} posts from initial load")

        # 滾動頁面觸發更多 API 請求
        for i in range(n_pages):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            if (i + 1) % 10 == 0:
                print(f"[crawl] scroll {i+1}/{n_pages}, total posts={len(captured_posts)}")

        print(f"[crawl] total captured posts: {len(captured_posts)}")

        # 過濾相關文章，進入抓內文
        articles_relevant = 0
        for post in captured_posts:
            title = post.get("title", "")
            excerpt = post.get("excerpt", "") or ""
            post_id = post.get("id")
            forum_alias = post.get("forumAlias", FORUM)

            if not is_relevant(title, excerpt):
                continue
            articles_relevant += 1
            post_url = f"https://www.dcard.tw/f/{forum_alias}/p/{post_id}"

            # 標題本身當 query
            title_clean = title.strip()
            if 8 <= len(title_clean) <= 60 and title_clean not in seen_queries:
                seen_queries.add(title_clean)
                results.append({
                    "qid": f"dcard_{qid_counter:04d}",
                    "query": title_clean,
                    "source": "dcard_title",
                    "url": post_url,
                })
                qid_counter += 1

            # 從 excerpt 抽 query
            for q in extract_queries(title + "\n" + excerpt):
                if q not in seen_queries:
                    seen_queries.add(q)
                    results.append({
                        "qid": f"dcard_{qid_counter:04d}",
                        "query": q,
                        "source": "dcard_excerpt",
                        "url": post_url,
                    })
                    qid_counter += 1

        # 對相關文章抓完整內文（限前 50 篇避免太慢）
        relevant_posts = [p for p in captured_posts if is_relevant(p.get("title",""), p.get("excerpt","") or "")]
        print(f"[crawl] relevant articles: {articles_relevant}, fetching content for top {min(50, len(relevant_posts))}...")

        content_captured: list[dict] = []

        def handle_content_response(response):
            url = response.url
            if "/api/v2/posts/" in url and "comments" not in url and "links" not in url:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "content" in data:
                        content_captured.append(data)
                except Exception:
                    pass

        page.on("response", handle_content_response)

        for post in relevant_posts[:50]:
            post_id = post.get("id")
            forum_alias = post.get("forumAlias", FORUM)
            post_url = f"https://www.dcard.tw/f/{forum_alias}/p/{post_id}"
            try:
                page.goto(post_url, wait_until="networkidle", timeout=20000)
                time.sleep(1.5)
            except Exception as e:
                print(f"  [warn] {post_id}: {e}")
                continue

            # 找到對應的內文
            if content_captured:
                detail = content_captured[-1]
                content = detail.get("content", "")
                for q in extract_queries(post.get("title","") + "\n" + content):
                    if q not in seen_queries:
                        seen_queries.add(q)
                        results.append({
                            "qid": f"dcard_{qid_counter:04d}",
                            "query": q,
                            "source": "dcard_content",
                            "url": post_url,
                        })
                        qid_counter += 1

        browser.close()

    print(f"\n[done] total queries extracted: {len(results)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=60, help="Number of scroll pages")
    ap.add_argument("--out", default="data/raw/dcard_real_queries.jsonl")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = crawl(args.pages, out_path)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[saved] {len(results)} queries → {out_path}")

    # Print sample
    print("\n=== Sample queries ===")
    for r in results[:10]:
        print(f"  [{r['source']}] {r['query']}")


if __name__ == "__main__":
    main()
