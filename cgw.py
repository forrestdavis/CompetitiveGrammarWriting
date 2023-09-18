from nltk import Nonterminal
from nltk.grammar import PCFG
from pcfg import PCFG as OtherPCFG
from nltk.parse import pchart

import math
import argparse
import sys

def loadGrammar(fnames: list[str], gType='parser') -> [PCFG | OtherPCFG]:  
    """
    Maps from grammar files to string for PCFG format.

    Arguments:
        fnames (list[str]): List of *.gr filenames
        gType (str): parser or generator

    Returns:
        str: Formatted string for PCFG.fromstring
    """
    # {lhs : {rhs : count}}
    RULES = {}
    for fname in fnames:
        with open(fname, 'r') as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                # Remove comments
                comment_idx = line.find('#')
                if comment_idx != -1:
                    line = line[:comment_idx].strip()

                if line == '':
                    continue

                rule = line.split()
                if len(rule) > 4:
                    raise ValueError(f"Error: more than two symbols in the "\
                                     f"right hand side at line {linenum}: "\
                                     f"{' '.join(rule)}")

                if len(rule) < 3:
                    raise ValueError(f"Error: unexpected line at {linenum}: "\
                                     f"{' '.join(rule)}")

                (count, lhs) = (int(rule[0]), rule[1])

                if 'Vocab.gr' in fname:
                    if len(rule) < 4: 
                        # Unary rule
                        rhs = f'"{rule[2]}"'
                    else:
                        rhs = " ".join(rule[2:])
                else:
                    rhs = " ".join(rule[2:])

                if lhs not in RULES:
                    RULES[lhs] = {}
                if rhs not in RULES[lhs]:
                    RULES[lhs][rhs] = 0
                RULES[lhs][rhs] += count

    # Normalize
    normalize(RULES)
    ruleString = toString(RULES)
    if gType == 'generator':
        return OtherPCFG.fromstring(ruleString)
    else:
        return PCFG.fromstring(ruleString)

def toString(rules: dict, start='TOP') -> str:
    """ From dictionary of rules to string.
     e.g., NP -> Det N [0.5] | NP PP [0.25] | "John" [0.1] | "I" [0.15]
    """

    string = ''
    for lhs in rules:
        rights = []
        for rhs in rules[lhs]:
            prob = rules[lhs][rhs]
            rights.append(f"{rhs} [{prob}]")
        if lhs == start:
            string = f"{lhs} -> {' | '.join(rights)}\n" + string
        else:
            string += f"{lhs} -> {' | '.join(rights)}\n"

    return string


def normalize(rules: dict) -> None:
    """ Inplace normalize by lhs rule. 
    """
    for lhs in rules:
        total = sum(rules[lhs].values())
        for rhs in rules[lhs]:
            count = rules[lhs][rhs]
            rules[lhs][rhs] = count/total

def parse(parser: pchart, line: str) -> list:
    """ Parse a sentence and return parses"""
    tokens = line.split()
    return list(parser.parse(tokens))

def getBestParse(parser: pchart, line: str): 
    """ Return best parse tree object """
    tokens = line.split()
    best = None
    for tree in parser.parse(tokens):
        if best is None:
            best = tree
        else:
            if best.prob() < tree.prob():
                best = tree
    return best

def perplexity(parser: pchart, sentences: list[str], verbose: bool = False) -> float:
    """Calculate approximate perplexity on list of sentences using the
    best parse"""

    denom = 0
    num = 0
    for sentence in sentences:
        tree = getBestParse(parser, sentence)
        if verbose:
            print(f"Input: {sentence}")
            print(f"Parse:")
            tree.pretty_print()
            print(f"Prob: {tree.prob()}")
        if tree:
            prob = tree.prob()
            if prob == 0:
                prob = 0.0000000000000000000001
                print(f"Sentence: {sentence} has 0 prob")
            num += -math.log2(prob)
            denom += len(sentence.split())
        else:
            prob = 0.0000000000000000000001
            num += 10000
            print(f"Error with: {sentence}")
            denom += len(sentence.split())

    return 2**(num/denom)

def crossEntropy(parser: pchart, sentences: list[str], verbose: bool = False) -> float:
    """Calculate approximate cross entropy on list of sentences using the
    best parse"""

    denom = 0
    num = 0
    for sentence in sentences:
        tree = getBestParse(parser, sentence)
        if verbose:
            print(f"Input: {sentence}")
            print(f"Parse:")
            tree.pretty_print()
            print(f"Prob: {tree.prob()}")
        num += math.log2(tree.prob())
        denom += len(sentence.split())
    return num/denom

def generate(generator:OtherPCFG, n: int) -> list[str]: 
    """Generate n number of sentences"""
    return list(generator.generate(n))

def main():

    flags = argparse.ArgumentParser(
        prog="python cgw.py", 
        description="Competitive Grammar Writing Tools", 
    )

    group = flags.add_mutually_exclusive_group()
    group.add_argument('-g', '--generate', help='set to generate', 
                        action='store_true')
    group.add_argument('-p', '--parse', help='set to parse', 
                        action='store_true')
    group.add_argument('-s', '--score', help='set to score using perplexity', 
                        action='store_true')
    flags.add_argument('-v', '--verbose', help='perplexity score is verbose', 
                        action='store_true')
    flags.add_argument('-n', '-num_sents', help='number of sentences to '\
                        'generate', type=int, default=20) 
    flags.add_argument('-i', '--input', help='grammar files', 
                       type=str, nargs='+')
    flags.add_argument('-f', '--file', type=argparse.FileType('r'), 
                        default=(None if sys.stdin.isatty() else sys.stdin))

    args = flags.parse_args()


    if args.parse:
        grammar = loadGrammar(args.input)
        parser = pchart.InsideChartParser(grammar)

        for line in args.file:
            line = line.strip()
            if line == '':
                continue
            tree = getBestParse(parser, line)
            print(f"Input: {line}")
            if tree is None:
                print("Parse: None")
                continue
            print("Parse:")
            tree.pretty_print()
            print(f"Prob: {tree.prob()}")

    elif args.generate:
        grammar = loadGrammar(args.input, gType='generator')
        for sentence in generate(grammar, args.n):
            print(sentence)

    elif args.score:
        grammar = loadGrammar(args.input)
        parser = pchart.InsideChartParser(grammar)

        sentences = []
        for line in args.file:
            line = line.strip()
            if line == '':
                continue
            sentences.append(line)
        score = perplexity(parser, sentences, args.verbose)
        print('-'*80)
        print(f'Score (lower is better): {score}')

main()
