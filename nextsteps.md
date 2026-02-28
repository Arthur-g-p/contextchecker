# What do I need? Regular runner. Check! Got it

was fehlt noch in der pipe
retry matrix
extractor klugscheißer (validator)
json saver!
ssl skipper

return original response in results_final too

# autocheck:

gets all the extractor and checker verdicts based on all test data. Reads the RESPONSE, QUESTION AND REFRENCE!
--> returns a file with response kg and verdics
but only runs on the msmacro data and other settings. missing that.

# eval (means just eval of the ACTUAL file (fachliche evaluation der ergebnisse, nicht der methodologie)):

uses the previous output of autocheck
sees the matching results. between human annotations and model results. however, i made a change to the code. when the model abstained, I it detected no hallucination.
Meine Annahme: wenn der extractor nichts extrahiert, dann nimmt das skript hallu rate 0%!
W I made the change!!
--> but the output?! is it EVER USED?

The output is a small hosn file with
avstain, entailment and neutral rates per setting and the avg.

So completely unnecssary.

# corr:

the actual meta evaluation. it looks at the numbers between abstention rates and looks for correlations!!!
here are many good numbers. so wie eine wetterrader das immer sonne anzeigt. hat 99% wahrscheinlichkeit aber nicht das was wir brauchen

I could run this complete eval pipeline to test a model from front to end. However: thats the meta eval. what about testing each seperatly in this eval script?
tract eval --> in paper: they tasked a model to add missing triplets did T/F label on existing ones. then looked for human correlation on this task.

however we are working here with a gt, which we want to utilize. Which is the reason for all the NLI CHecks to find best prompts

checker eval --> comparing verdics to gt
metaeval --> what we just said

übrigens: div by zero crash.

# extraction saver

# prompts with DSPY

# evaluation scripts

# eval comparison

# integrating with frontend

# include RAGCHECKING

# Write article

# änderungen mitteilen, exhaustive

# build two applications on it

ketchUP
live contradiction checker

lazy iimport big dependecies?

# error handling:

even warns you on cache hits!!!

parsing error will get catched by OPENAI SDK!! i might need to prevent it!

--> redo failed parsing calls. re-run from caller. Maybe slight variation of prompt?

I want to release it using CLI. when ppl keyboard interrupt can I catch the event and make a graceful shutdown? this increases quality of life by a lot imo

if no schema -- different prompt is needed!!

cli version i feel like most go with envs. but does it work for me?


Exact numbers game!!! close to etc may not reap results you expect. exact number mode?


if json parsing fails multiple times we will send out solo extraction..



metaeval needs to make an entire run!



strange errors: #API ERROR (openrouter/qwen/qwen3.5-35b-a3b) — Attempt 1/3: litellm.APIError: APIError: OpenrouterException - Upstream error from Alibaba: <400> InternalError.Algo.InvalidParameter: 'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.



set timeout... make is somewhere settable. need a proper config first of all. timeout errors are retries. 

make a rerun concept


What about a new value I could propose: precision of the answer? 

caching strategy: it reads empty entries and dynmically adds them to batch. the pipe stays dumb.