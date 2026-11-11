from __future__ import annotations

from app.kb import Article, KnowledgeBase, tokenize


def test_loads_all_ten_articles(kb: KnowledgeBase) -> None:
    assert len(kb.articles) == 10
    assert all(article.title and article.text for article in kb.articles)


def test_finds_the_relevant_article_first(kb: KnowledgeBase) -> None:
    assert kb.search("refund annual plan")[0].file == "refunds.md"
    assert kb.search("forgot password reset email not arriving")[0].file == "reset-password.md"
    assert kb.search("API 429 rate limit")[0].file == "api-rate-limits.md"
    assert kb.search("lost phone authenticator recovery codes")[0].file == "two-factor-authentication.md"


def test_results_are_ranked_and_limited(kb: KnowledgeBase) -> None:
    hits = kb.search("billing invoice charge", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score > 0


def test_no_match_and_empty_query_return_nothing(kb: KnowledgeBase) -> None:
    assert kb.search("zzzz qqqq") == []
    assert kb.search("") == []
    assert kb.search("the and of") == []


def test_rare_terms_outweigh_common_ones() -> None:
    kb = KnowledgeBase(
        [
            Article("a.md", "Plans", "plan plan plan pricing"),
            Article("b.md", "Plans", "plan export"),
            Article("c.md", "Plans", "plan seats"),
        ]
    )
    assert kb.search("plan export")[0].file == "b.md"


def test_tokenize_drops_stop_words_and_keeps_unicode() -> None:
    assert tokenize("How do I reset my contraseña?") == ["reset", "contraseña"]
