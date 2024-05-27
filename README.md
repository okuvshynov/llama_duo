# llama duo - asyncronous/distributed speculative decoding for llama3. 

llama duo is an attempt to make simple linear speculative decoding work in parallel with the main model. It is mostly intended to work in situations when two devices are available (e.g. Mac Mini and laptop) and we attempt to use the second device to speed up the generation.
Not every hardware/model combination would benefit from such setup. 

Example of the configuration which gets good speedup:
Apple M1 (16GB RAM) runs Llama3-8B-Instruct @ Q8 and Apple M2 (24GB RAM) runs Llama3-8B-Instruct @ Q4.

Example of configuration which doesn't get much value:
Apple M1 (16GB RAM) + Apple M2 Ultra (192GB RAM). M2 Ultra is order of magnitude faster and second model is unable to keep up.

The important potential appliaction for this approach would be to use speculation to speed up evaluation of huge models (e.g. hopefully upcoming llama3-405B), when the main model itself will be split between multiple devices and the 'spare compute' they would have will be used to speculate remotely.
more plans on this https://github.com/okuvshynov/llama_duo/issues/1 and https://github.com/okuvshynov/llama.cpp/tree/duo/examples/duo


## Dependencies

1. [llama.cpp](https://github.com/ggerganov/llama.cpp)
2. [nlohmann/json](https://github.com/nlohmann/json)
3. [cpp-httplib](https://github.com/yhirose/cpp-httplib)

dependencies are being pulled using cmake FetchContent, so there's no need to install these libraries manually.

For the CLI chat.py, needs python and requests module.

## Installation

```
mkdir _build && cd _build
cmake ..
make -j 4
pip install requests
```

After this step you should have two binaries built: ```lead``` and ```back```. 

## Distributed example

On M2 Macbook with 24 GB memory start ```lead``` service with full fp16 precision 8B model:

```
./lead -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-fp16.gguf -ngl 99
```

On M1 Mini with 16GB memory start ```back``` service and specify the ```lead``` host:

```
./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct.Q3_K_M.gguf --host 169.254.226.241 -ngl 99
```

Both of these services will evaluate model on GPU (```-ngl``` flag). The model they run is essentially the same, except smaller and slower machine runs more aggressively quantized version.

Now on the macbook start the chat and ask a question:

```
python chat.py
You: Illustrate the difference between concurrency and parallelism in python.
```

```
...
I: decoded  737 tokens in   81.129 seconds, speed:    9.084 t/s
I: total generation time: 100.386
```

Running same model without speculation is much slower:

```
..
I: decoded  737 tokens in  222.631 seconds, speed:    3.306 t/s
I: total generation time: 224.635
```

## Inverted distributed example

We use the same M2 Macbook and M1 Mini, but ```lead``` now runs on Mac Mini:

```
./lead -ngl 99 -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q8_0.gguf
```

chat and ```back``` run on the laptop:

```
./back -ngl 55 -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --host 169.254.90.21
```

```
python chat.py http://169.254.90.21:5555
You: Implement a simple lock-free container in c++
```

Even though M2 has better GPU and more unified RAM, such setup was useful as resources on the laptop are needed for other applications as well, like keeping a few Chrome tabs open.

With async speculation:
```
I: encoded  104 tokens in    1.181 seconds, speed:   88.026 t/s
...
I: decoded  695 tokens in   52.814 seconds, speed:   13.159 t/s
I: total generation time: 53.9993
```

Without async speculation:
```
I: encoded  104 tokens in    1.270 seconds, speed:   81.912 t/s
...
I: decoded  692 tokens in  103.642 seconds, speed:    6.677 t/s
I: total generation time: 104.914
```

You can not a different number of tokens in async speculation - that happened because we evaluated sequence and got eot in the middle of it.

## Local example

Here we run it on single M2 Ultra, using GPU for main model and CPU for second model. 

Start lead with Llama3-70B@Q8 model with all layers on GPU and default settings for interface/port 0.0.0.0:5555:

```
./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99
```

Start back with Llama3-8B@Q4 model on 16 CPU threads. It looks for lead service on localhost:5555 by default.

```
./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --n-gpu-layers 0 --threads 16
```

Start basic chat command-line interface (also defaults to localhost:5555):

```
python chat.py
```

In chat window ask the model something: 

```
You: Illustrate the difference between concurrency and parallelism in python.
```

What we should observe:

1. ```lead``` service should start printing out the generated tokens, highlighing accepted tokens in green.
2. ```back``` would print some debug info.
3. After the generation is complete, the response would be returned to chat.


```lead``` would print out some timing info:

```
I: encoded  105 tokens in    3.108 seconds, speed:   33.786 t/s
...
I: decoded  784 tokens in   75.159 seconds, speed:   10.431 t/s
I: total generation time: 78.2696
```

## Simulating failures

Note that ```back``` service is optional - we can turn it off, run the main model as before:

```
./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99
```

```
python chat.py
```

In chat window ask the same question: 

```
You: Illustrate the difference between concurrency and parallelism in python.
```

And observe the same output.

```
I: encoded  105 tokens in    2.699 seconds, speed:   38.908 t/s
...
I: decoded  784 tokens in   92.639 seconds, speed:    8.463 t/s
I: total generation time: 95.3407
```

As we can see, it is slower.


We can also start/stop/simulate non-availability/failure for ```back``` service. As in previous example, start main model and chat:

```
./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99
```

```
python chat.py
```

In chat window ask the model the same question: 

```
You: Illustrate the difference between concurrency and parallelism in python.
```

At some moment during generation start the ```back``` service:

```
./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --n-gpu-layers 0 --threads 16
```

```back``` service would catch up with ```lead``` by processing input prompt + the tokens generated to this point and start speculating.
The performance would be somewhere in between the two runs above

```
I: encoded  105 tokens in    2.765 seconds, speed:   37.969 t/s
...
I: decoded  784 tokens in   82.254 seconds, speed:    9.568 t/s
I: total generation time: 85.0213
```

We can also kill the back service sometime in the middle of query processing, start it again, etc.

## How it works

It is simple linear speculation, except it is generated in parallel with main model and reconciled after each lead token generation.

We can think of three separate sequences:
1. local sequence on ```lead``` -- this is ground truth, which will be equivalent to main model producing tokens one by one. Let's call this sequence ```L```.
2. local sequence on ```back``` -- this is the speculated sequence which we work on in parallel. Let's call this sequence ```B```.
3. shared speculation sequence on ```lead``` -- it serves as a communication channel between ```lead``` and ```back``` models. Let's call this sequence ```S```. This sequence might contain tokens of two types: ```approved```, which were confirmed by main model and are also part of ```L``` and ```not_rejected``` - produced by speculation model, but we don't know yet if it will be approved or not.

Let's look at the following example:

```lead``` got a request from chat.py with prompt ```[The, quick, brown]```. Sequences ```L``` and ```S``` are initialized with it.

```lead``` and ```back``` start working on it in parallel. All operations involving read/write from/to ```S``` are guarded with mutex so that lead and back would not modify it simultaneously. Let's consider the following event sequence.

0. Initialization
<pre>
L = [the, quick, brown]
B = []
S = [<b>the, quick, brown</b>]
</pre>

1. ```back``` calls ```lead``` periodically to check if there's some work. If yes, set ```B := S```
<pre>
L = [the, quick, brown]
B = [the, quick, brown]
S = [<b>the, quick, brown</b>]
</pre>

2. ```back``` produces 'fox'.
<pre>
L = [the, quick, brown]
B = [the, quick, brown, fox]
S = [<b>the, quick, brown</b>]
</pre>

3. ```back``` calls ```lead``` and compares ```B``` with ```S```. 'fox' and appended to the ```S``` in 'not_rejected' state.
<pre>
L = [the, quick, brown]
B = [the, quick, brown, fox]
S = [<b>the, quick, brown</b>, fox]  
</pre>

4. ```back``` produces 'jumps'.
<pre>
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps]
S = [<b>the, quick, brown</b>, fox]  
</pre>


5. ```back``` calls ```lead``` and compares ```B``` with ```S```. 'jumps' is appended to ```S``` in 'not_rejected' state.
<pre>
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps]
S = [<b>the, quick, brown</b>, fox, jumps]
</pre>


6. ```back``` produces 'into'.
<pre>
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps, into]
S = [<b>the, quick, brown</b>, fox, jumps]
</pre>


7. ```back``` calls ```lead``` and compares ```B``` with ```S```. 'into' is appended to ```S``` in 'not_rejected' state.
<pre>
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps, into]
S = [<b>the, quick, brown</b>, fox, jumps, into]
</pre>


8. ```lead``` produces 'fox'. 'fox' is appended to ```L```.
<pre>
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into]
S = [<b>the, quick, brown</b>, fox, jumps, into]
</pre>


9. ```lead``` compares ```L``` with ```S```. As 'fox' matches, it is marked is approved, 'jumps into' stays not_rejected, main model starts working on input of 3 tokens 'fox jumps into'.
<pre>
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into]
S = [<b>the, quick, brown, fox</b>, jumps, into]
</pre>


10. ```back``` produces 'the'.
<pre>
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the]
S = [<b>the, quick, brown, fox</b>, jumps, into]
</pre>


11. ```back``` calls ```lead``` and compares ```B``` with ```S```. 'the' is appended to ```S``` in 'not_rejected' state.
<pre>
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the]
S = [<b>the, quick, brown, fox</b>, jumps, into, the]
</pre>


12. ```back``` produces 'big'.
<pre>
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [<b>the, quick, brown, fox</b>, jumps, into, the]
</pre>


13. ```back``` calls ```lead``` and compares ```B``` with ```S```. 'big' is appended to ```S``` in 'not_rejected' state.
<pre>
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [<b>the, quick, brown, fox</b>, jumps, into, the, big]
</pre>


14. ```lead``` produces 'jumps over the'. First, we need to compare the output with input (in this case, 'fox jumps into'). As 'jumps' matches, but 'over' != 'into', we accept 'jumps over' and append it to ```L```. We cannot accept 'the', because it was produced as an continuation to the sequence 'the quick brown fox jumps into', and we now know that 'into' was wrong.
<pre>
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [<b>the, quick, brown, fox</b>, jumps, into, the, big]
</pre>


15. ```lead``` compares L with S. We reject 'into the big', remove them from the sequence ```S``` and assign ```S := L```. ```lead``` works on a single input 'over'.
<pre>
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [<b>the, quick, brown, fox, jumps, over</b>]
</pre>


16. ```back``` produces 'puddle'.
<pre>
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, into, the, big, puddle]
S = [<b>the, quick, brown, fox, jumps, over</b>]
</pre>


17. ```back``` calls lead and compares ```B``` with ```S```. We see a mismatch, append nothing to ```S```, and assign ```B := S```.
<pre>
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, over]
S = [<b>the, quick, brown, fox, jumps, over</b>]
</pre>

The actual implementation is a little more complicated because:
1. communication between ```lead``` and ```back``` involves passing delta rather than entire sequence - otherwise we'd end up with large messages for long contexts.
2. ```back``` needs to support starting in the middle of processing of main model.

It's probably best to check the code to see the details.

## Comparison with synchronous speculative decoding 

Test set up:
1. Same hardware - M2 Ultra
2. Software - [speculative](https://github.com/ggerganov/llama.cpp/tree/master/examples/speculative) from llama.cpp with same models
3. Same formatted prompt:

prompt:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Illustrate the difference between concurrency and parallelism in python.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

4. Command:
```
./speculative -m ../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf -md ../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf -f /tmp/p.txt -e -ngl 99 -t 4 -n 1024 -c 4096 -s 8 --top_k 1 -ngld 99
```

Results in decoding speed of 8.496 t/s, which is somewhere in between async speculation and no speculation.

## limitations
* llama3 instruct hardcoded prompt format.
* only tested on Apple devices (M1, M2, M2 Ultra).
* greedy sampling

## Effectiveness of speculative evaluation

Regardless of sync/async, speculative evaluation has different effectiveness for difference hardware/model/quantization level combinations.

See some discussion here: https://github.com/ggerganov/llama.cpp/discussions/6777

And as another datapoint - fp16 Llama3-70B on M2 Ultra would have difference characteristics. 

<img align="middle" width="782" alt="Screenshot 2024-05-16 at 11 07 36 AM" src="https://github.com/okuvshynov/experiments/assets/661042/7cb29ebb-9aed-4a75-98fb-de2792804d6f">



## TODO

```
[ ] Both async and sync speculation - if we don't have good candidate, generate N new tokens in place.
[ ] Tree-based speculation
[ ] beam search, not greedy sampling only.
[ ] make it work with some popular UI/API (what are those?)
[ ] No hardcoded models
[ ] Saving cache between sessions.
[ ] Hardware to try it on:
  [ ] something small -  Raspberry Pi + Phi model
  [ ] large CPU-only servers with a lot of RAM.
  [ ] iPhone/iPad for chat + speculation model?
```
