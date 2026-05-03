from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchAny

client = QdrantClient(host='localhost', port=6333)

query_filter = Filter(must=[FieldCondition(key="Frame", range=Range(gte=100, lte=5000))])

res = client.scroll(
    collection_name='images',
    scroll_filter=query_filter,
    limit=5,
    with_payload=True
)
print("With Frame range filter:")
for p in res[0]:
    print(p.payload)
