# API rate limits

The Tasklane REST API allows 120 requests per minute per token on Pro and 600 per minute on Business. When the limit is exceeded the API returns HTTP 429 with a Retry-After header in seconds. Back off and retry after that time.

Bulk endpoints accept up to 100 items per request and count as one request. Higher limits are available on Enterprise plans by request.
