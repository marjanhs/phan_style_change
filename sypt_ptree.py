from pycorenlp import StanfordCoreNLP
from nltk.tree import ParentedTree
import re, os, subprocess
from nltk.tokenize import sent_tokenize, word_tokenize
arrow = ' => '


def leaf_to_root(node, ignoreleaf=True):
    rules=[]
    while node.label()!="ROOT":
        rule= node.label() + arrow
        for s in node:
            if type(s)== ParentedTree:
                rule += s.label() +' '
            else:
                rule += s +' '
        rules.append(rule)
        node = node.parent()
    if ignoreleaf and len(rules) >1:
        return rules[1:]
    else:
        return rules


def traverse(parsed_sentence, ignoreleaf=True):

    t = ParentedTree.fromstring(parsed_sentence)
    q = [t]
    rules = []
    while len(q) > 0:
        current = q.pop()
        for s in current:
            if type(s) == ParentedTree:
                q.append(s)
            else:
                rules.append(leaf_to_root(current, ignoreleaf=ignoreleaf))

    return rules


def get_pt_features_coreNLP(doc, ignoreleaf=True):
    en = doc.encode('utf-8')
    de = en.decode('utf-8')
    doc = de
    chars_to_remove = ['{', '}', '(', ')']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    doc = re.sub(rx, '', doc)
    nlp = StanfordCoreNLP('http://localhost:9000')
    sentences = sent_tokenize(doc)
    ptree_features = list()
    for sentence in sentences:
        try:
            if sentence != "" and len(word_tokenize(sentence)) <= 80: # less than 50 words
                output = nlp.annotate(sentence, properties={
                    'annotators': 'parse',
                    'outputFormat': 'json'
                })

                parsed = (output['sentences'][0]['parse'])
                rules = traverse(parsed, ignoreleaf=ignoreleaf)
                ptree_features.append(rules)
        except:
            print('Problem in parsing sentece = %s' % sentence)

    return ptree_features


def get_pt_features_standalone(doc, tmp_path='', ignoreleaf=True):

    tmp_path = os.path.join(tmp_path, 'tmp_marjan')

    args = ['java','-mx3000m', '-cp', 'stanford-parser-full-2018-02-27/stanford-parser.jar',
            'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
            '-encoding', 'utf-8',
            '-model', 'stanford-parser-full-2018-02-27/englishPCFG.ser.gz',
            '-maxLength', '50',
            '-sentences', 'newline',
            '-outputFormat', 'penn',
            tmp_path]

    en = doc.encode('utf-8')
    de = en.decode('utf-8')
    doc = de
    chars_to_remove = ['{', '}', '(', ')']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    doc = re.sub(rx, '', doc)

    sentences = sent_tokenize(doc)
    ptree_features = list()

    with open(tmp_path, 'wt') as fw:
        for sentence in sentences:
            if sentence.strip() != '':
                fw.write(sentence+'\n')

    p = subprocess.Popen(args, stdin=None, stdout=-1, stderr=-1)
    outs, err = p.communicate()
    outs = outs.decode('utf-8').replace('(())\n', '')  # removing output of long sentences
    for parsed in outs.split('\n\n'):
        if parsed != "":
            try:
                rules = traverse(parsed, ignoreleaf= ignoreleaf)
                ptree_features.append(rules)
            except ValueError:
                print('Problem in converting parsed sentence =  %s ' % (parsed))
                raise

    return ptree_features