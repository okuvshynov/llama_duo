curl -s -S -X                         \
  POST "http://127.0.0.1:5555/query"  \
  -H 'Content-Type: application/json' \
  -d '{"text": "What is the difference between ", "offset": 0, "complete": false}'

sleep 2

curl --no-buffer -s -S -X                         \
  POST "http://127.0.0.1:5555/query"  \
  -H 'Content-Type: application/json' \
  -d '{"text": "What is the difference between concurrency and parallelism", "offset": 0, "complete": true}'

