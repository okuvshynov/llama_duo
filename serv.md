### Command

```
./serv -m ../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --top_p 0.0 --top_k 1
```

### How it should work?

1. We maintain a list of sessions. One of the sessions might be active at a time. Each session is associated with evaluation state (sequence)
2. None of the sessions might be active at a given moment.
3. When request comes, it might be for active session, not active session or new request (so we need new session context)
4. We prioritize requests based on the 'completed' - if we need to generate something for client right away, we should do that. So priorities are: current active request, next 'completed' requests, FIFO requests which do prefetch. 
5. We can still merge tokens in multiple threads, but there's single thread for doing heavy lifting (llama\_decode)
6. What data structures do we need? thread-safe session\_id -> session\_context map. We can just iterate over that map to see if we have any active session to work on.
