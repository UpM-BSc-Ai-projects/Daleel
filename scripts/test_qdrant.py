import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

client = QdrantClient(host='localhost', port=6333)
print('Total points:', client.count('images').count)

# Get some points to see payload
res, _ = client.scroll(collection_name='images', limit=2)
for p in res:
    print('Payload:', p.payload)

# Try a filter
f = Filter(must=[FieldCondition(key='Cam', match=MatchAny(any=['Live']))])
res, _ = client.scroll(collection_name='images', limit=2, scroll_filter=f)
print('Filtered MatchAny Live:', len(res))

f2 = Filter(must=[FieldCondition(key='Cam', match=MatchValue(value='Live'))])
res, _ = client.scroll(collection_name='images', limit=2, scroll_filter=f2)
print('Filtered MatchValue Live:', len(res))
