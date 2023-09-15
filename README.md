# Competitive Grammar Writing

The following is a minimal implementation of the competitive grammar writing
assignment detailed in [Eisner and Smith
(2008)](https://aclanthology.org/W08-0212/). I additionally draw on Stanford's
assignment ([link](https://nlp.stanford.edu/~socherr/CGWDocumentationCS224N))
and the grammar files from Prof. [Anoop Sarkar](http://anoopsarkar.github.io/)'s
competitive grammar writing assignment (write up
[link](http://anoopsarkar.github.io/nlp-class/cgw.html); github
[link](https://github.com/anoopsarkar/cgw)). Additionally,
[nltk](https://github.com/nltk/nltk) is used to encode the grammar and parse the
sentences, and the [pcfg](https://github.com/thomasbreydo/pcfg) package. 
I am indebted to all these resources! 

## Quick Start 

    git clone https://github.com/forrestdavis/CompetitiveGrammarWriting.git
    pip install -r requirements.txt

## Task Overview

In this exercise, you'll write a PCFG for a small subset of English. Unlike the
assignments we've done so far, this won't involve any traditional programming.
Instead, you will create your PCFG as a set of weighted rules and write them as
tab-separated values. We've also provided you with tools that can evaluate the
performance of your PCFG.  

Your grammar should be sophisticated enough to parse
a variety of English sentences, but it should not produce any ungrammatical
output. Since this is a probabilistic CFG, your grammar should also give more
probability mass to likely sentences and less probability mass to unlikely ones.
We have a sample set of sentences you can use to develop your grammar and test
how much it can cover.  

Of course, this is competitive (yet, friendly) grammar writing. Your
grammar should be able to generate sentences that other groups' grammars can't
handle. And likewise, it should be prepared to handle as many grammatical
sentences from other teams' grammars as possible.

### Setup

You will be working in a group of 3-4. As a warm up, please ensure you are able
to run the commands below. A notebook version of this can be found in
`cgw.ipynb'. If you prefer working in the cloud, you can find a colab version
[here](https://colab.research.google.com/drive/1EvmQlr4gM7rQ9nagBEViYs1o1E212Yd3?usp=sharing). 

### Language

As mentioned, you'll create a grammar to describe a subset of English: your
sentences must use the words listed in the `allowed_words.txt` file. You are not
allowed to add new words to the vocabulary. Any sentence that can be produced
with these words is fair game.

If you take a look through the file, you'll see hundreds of words from various
linguistic categories. (All capitalized words are proper nouns.) The more of
these you can accurately handle, the better your grammar will be. You are
encouraged to add other part of speech tags for the existing words (e.g.`run`
can be a verb or an adjective).

### Creating a PCFG

#### PCFG overview

A probabilistic context-free grammar consists of:

* A set of non-terminal symbols
* A set of terminal symbols
* A set of rewrite or derivation rules, each with an associated probability
* A start symbol

For natural language PCFGs, we think of the start symbol as indicating
"sentence" (in this case it will be TOP), and the terminal symbols as the words.
A derivation rule gives one way to rewrite a non-terminal symbol into a sequence
of non-terminal symbols and terminal symbols. For example, `S -> NP VP` says
that an `S` (perhaps indicating a declarative sentence) can be rewritten as an
`NP` (noun phrase) followed by a `VP` (verb phrase).

#### Files

To create your PCFG, you will be creating and editing grammar files. These all
end in the suffix .gr. We have provided three for you already:

* S1.gr
This is a starter grammar that contains a few simple rules. It generates real
English sentences, but it's very limited.

* S2.gr
This is a weighted context-free grammar that generates all possible sentences.
If you could design S1 perfectly, then you wouldn't need S2. But since English
is complicated and time is short, S2 will serve as your backoff model. (For
details, see the Appendix.)

* Vocab.gr
This gives a part-of-speech tag to every word in `allowed_words.txt`. Feel free
to change these tags or create new ones. You will almost certainly want to
change the tags in rules of the form `Misc -> word`. But be careful: you don't
want to change `Misc -> goes` to `VerbT -> goes`, since goes doesn't behave like
other `VerbT`'s. In particular, you want your `S1` to generate `Guinevere has
the chalice .` but not `Guinevere goes the chalice .`, which is ungrammatical.
This is why you may want to invent some new tags.

#### Derivation rules

All derivation rules have the same structure:

    <weight> <parent> <child1> <child2> ...

`<weight>` is an integer value. Weights allow you to express your knowledge of
which English phenomena are common and which ones are rare. By giving a low
weight to a rule, you express a belief that it doesn't occur very often in
English.

In a probabilistic CFG, each rule has some probability associated with it, and
the probability of a derivation for a sentence would be the product of all the
rules that went into the derivation. We don't want you to worry about making
probabilities sum up to one, so you can use any positive number as a weight. We
renormalize them for you so that the weights for all the rules which rewrite a
given non-terminal form a valid probability distribution.

`<parent>` is a non-terminal symbol. All children (e.g. `<child1>`) are either
terminals or non- terminals. All rules are at most binary branching. 

The parser will basically treat any symbols for which the grammar contains no
rewrite rules as terminals and throw an error if it finds any such terminals not
in the allowed words list. Also note that lines beginning with the `#` symbol
are considered comments. You can give any of the utilities a set of files
containing such rules, and they will merge the rules in each file into a single
grammar.

You can change these files however you like, and you can create new ones, too.
However, you **must** not create new vocabulary words (terminals).

#### Weighting S1 and S2

Two rules your grammar **must** include are `TOP -> S1` and `TOP -> S2`. (By
default, these rules are in `S1.gr`.) The relative weight of these determines
how likely it is that `S1.gr` (with start symbol `S1`) or `S2.gr` (with start
symbol `S2`) would be selected in generating a sentence, and how costly it is to
choose one or the other when parsing a sentence.

Choosing the relative weight of these two rules is a gamble. If you are
over-confident in your "real" English grammar (`S1`), and you weight it too
highly, then you risk assigning very low probability to sentences which `S1`
cannot generate (since the parser will have to resort to your `S2` to get a
parse, which gives every sentence a low score).

But if you weight `S2` too highly, then you will probably do a poor job of
predicting the test set sentences, since `S2` will not make any sentences very
likely. (It accepts everything, so probability mass is spread very thin across
the space of word strings.) Of course, you can invest some effort in trying to
make `S2` a better n-gram model, but that's a tedious task and a risky
investment.

Hint: If you have multiple laptops you can create more grammar files which will
eventually all be concatenated.

## Testing your PCFG

We've provided you with three tools for developing your PCFG. First, you can
test how well your PCFG can parse other sentences. Second, you can test how well
it can generate sentences of its own. Finally you can evaluate your models
ability to fit sentences (via perplexity). 

### Parsing Sentences

This command takes in a sentence file and a sequence of grammar files and parses
each of the sentences with the grammar. It will print out the maximum
probability parse tree and the probability for that parse. 

    python cgw.py -p -i *.gr < <sentences>

where `<sentences>` is a sentence file. 

Development data is included with the repository (`example_sentences.txt`). You
can see the sentences, their parses, and probabilities by running: 

    python cgw.py -p -i *.gr < example_sentences.txt

### Generating Sentences

You can generate sentences from your grammar with the following command: 

    python cgw.py -g -n <n> -i *.gr

where `<n>` is the number of sentences to generate.

This program takes a sequence of grammar files and performs a given number of
repeated expansions on the START symbol. This can be useful for finding glaring
errors in your grammar, or undesirable biases. For example, 

    >>> python cgw.py -g -n 20 -i *.gr
    another quest has any quest into another defeater
    a defeater is this fruit
    no sovereign carries that master .
    each swallow drinks Zoot
    the chalice of Dingo carries each husk
    another pound has every fruit .
    Arthur drinks that coconut
    any defeater below the land drinks every horse into each fruit .
    Patsy drinks that swallow on any land
    the defeater covers each master
    this servant is no coconut .
    that horse on any sun through Uther Pendragon drinks Sir Lancelot .
    that castle is any horse
    this weight carries each castle
    another winter has a sun .
    Sir Bedevere rides that corner .
    another fruit is this coconut
    this corner rides Guinevere .
    Uther Pendragon drinks another fruit .
    no defeater drinks each coconut


For our purposes, generation is just repeated symbol expansion. To expand a
symbol such as `NP`, our sentence generator will randomly choose one of your
grammar's `NP -> ...` rules, with probability proportional to the rule's weight.

### Evaluating your Grammar 

You can evaluate your grammars ability to predict sentences with the following
command: 

    python cgw.py -s -i *.gr  < <sentences>

where `<sentences>` is a sentence file. 

The perplexity of your grammar on the given sentence file is computed. The lower
the perplexity value, the better your grammar is at predicting the data. For
example, running the initial grammar on `example_sentences.txt` yields: 

    >>> python cgw.py -s -i *.gr  < example_sentences.txt
    --------------------------------------------------------------------------------
    Score (lower is better): 1152.1487753459742

## Submitting your files

Periodically in lab, you will be asked to submit your grammar files via Moodle.
Only one group member should submit. You can submit multiple times, and we will
make use of this to see which team is pulling ahead :smile:.

## Final Evaluation of Sentences

In the final minutes of lab, each group will be asked to evaluate a sentences
for grammaticality. You will be given a file with sentences randomly generated
from everyone's grammar. Delete the sentences you think are ungrammatical and
upload the file to Moodle.  

## Final Grade and Winner

Your final grade is determined by 1) whether the sentences your grammar
generates are grammatical, and 2) participation in the competition. The winning
team will be announced later in the week with a prize going to the best team.
