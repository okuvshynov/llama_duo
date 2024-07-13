curl --no-buffer -s -S -X                         \
  POST "http://127.0.0.1:5555/query"  \
  -H 'Content-Type: application/json' \
  -d @data/in0_0.json

# simulate some finishing typing
sleep 10

curl --no-buffer -s -S -X                         \
  POST "http://127.0.0.1:5555/query"  \
  -H 'Content-Type: application/json' \
  -d @data/in0.json

