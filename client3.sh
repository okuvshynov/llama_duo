curl --no-buffer -s -S -X                         \
  POST "http://127.0.0.1:5555/query"  \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "system", "content": "You are a helpful, respectful and honest assistant"}, {"role": "user", "content": "What is the difference between concurrency and parallelism"}], "complete": true}'

