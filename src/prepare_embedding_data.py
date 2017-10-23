# coding=utf-8
import pickle

from liner2 import Liner2, LinerWordnet
from optparse import OptionParser
import sys, codecs
import corpus2


def lemstrings_of_token(tok, first_lex_only=True):
    if first_lex_only:
        # forcing first disamb lexeme only
        for lex in tok.lexemes():
            if lex.is_disamb():
                return unicode(lex.lemma())
        # assert False, 'no disamb lemma here'
        return ''
    lemmas = set(
        unicode(lex.lemma())
        for lex in tok.lexemes()
        if lex.is_disamb())
    return lemmas

def orth_of_token(tok):
    return unicode(tok.orth())

def tagstrings_of_token(tok, tagset, expand_optional=True):
    tags = [lex.tag() for lex in tok.lexemes() if lex.is_disamb()]
    if not tags:
        return ''
    # forcing first disamb lexeme only
    tags = tags[:1]
    if expand_optional:
        # create multivalue tag wherever a value of optional attr is unspec
        tags = [tagset.expand_optional_attrs(tag) for tag in tags]
        # now expand multivalue tags to singular tags
        newtags = []
        for tag in tags:
            newtags.extend(tagset.split_tag(tag))
        tags = newtags
    return tagset.tag_to_string(tags[0])


def class_of_token(tok, tagset):
    ctag = tagstrings_of_token(tok, tagset)
    return unicode(ctag.split(':', 1)[0])



def go():
    parser = OptionParser(usage="Tool for preparing data for embeddings training")
    parser.add_option('-i', '--input-format', type='string', action='store',
                      dest='input_format', default='xces',
                      help='set the input format; default: xces')
    parser.add_option('-t', '--tagset', type='string', action='store',
                      dest='tagset', default='nkjp',
                      help='set the tagset used in input; default: nkjp')
    parser.add_option('-p', '--par-sep', type='string', action='store',
                      dest='par_sep', default='',
                      help='set the paragraph separator; default: (two newlines)')
    parser.add_option('-s', '--sent-sep', type='string', action='store',
                      dest='sent_sep', default='\n',
                      help='set the sentence separator; default: (newline)')
    parser.add_option('--separate-tokens', action='store_true', default=True,
                      dest='separate_tokens', help='separate all tokens with space')
    parser.add_option('--ignore-ns-sent', action='store_true', default=False,
                      dest='ignore_ns_sent', help='ignore no-space markers on sent boundaries')
    parser.add_option('-f', '--feature', type='string', action='store',
                      dest='feature', default='lemma',
                      help='feature; ,  default: (lemma)')
    parser.add_option('-w', '--wordnet_path', type='string', action='store',
                      dest='wordnet_path',
                      default='/home/michal/dev/ipi/korpusy/plwordnet_2_1_0/plwordnet_2_1_0_pwn_format',
                      help='wordnet path')
    parser.add_option('-o', '--output-format', type='string', action='store',
                      dest='output_format', default='text',
                      help='set the output format (text); default: text')
    parser.add_option('-l', '--limit', type='int', action='store',
                      dest='limit', default=0,
                      help='set the tokens number limit; default: 0')
    parser.add_option('--liner_jar', type='string', action='store',
                      dest='liner_jar',
                      default='/home/michal/dev/ipi/liner2/g419-liner2-cli/build/libs/g419-liner2-cli-2.5-SNAPSHOT-all.jar',
                      help='liner jar path, required for wordnet features')
    parser.add_option('--liner_lib', type='string', action='store',
                      dest='liner_lib',
                      default='/home/michal/dev/ipi/liner2/lib',
                      help='liner lib path, required for wordnet features')
    (options, args) = parser.parse_args()
    if len(args) != 2:
        print 'Need to provide input and output.'
        print 'See --help for details.'
        print
        sys.exit(1)
    fn_input, fn_output = args

    def get_wordnet():
        return LinerWordnet(options.wordnet_path, options.liner_jar, options.liner_lib)

    feature_generator = None
    if options.feature.startswith("hypernym"):
        wordnet = get_wordnet()
        feat = wordnet.get_hypernym_feature(options.feature, int(options.feature.split("-")[1]))
        def feature_generator(tok): return unicode(feat.generate(lemstrings_of_token(tok)))
    if options.feature.startswith("synonym"):
        wordnet = get_wordnet()
        feat = wordnet.get_synonym_feature()
        def feature_generator(tok): return unicode(feat.generate(lemstrings_of_token(tok)))
    if options.feature == 'ctag':
        def feature_generator(tok): return tagstrings_of_token(tok, tagset)
    if options.feature == 'orth':
        def feature_generator(tok): return orth_of_token(tok)
    if options.feature == 'lemma':
        def feature_generator(tok): return lemstrings_of_token(tok, tagset)
    if options.feature == 'lemma.ctag':
        def feature_generator(tok): return lemstrings_of_token(tok, tagset)+'.'+tagstrings_of_token(tok, tagset)
    if options.feature == 'lemma.class':
        def feature_generator(tok): return lemstrings_of_token(tok, tagset)+'.'+class_of_token(tok, tagset)
    if options.feature == 'class':
        def feature_generator(tok): return class_of_token(tok, tagset)

    if not feature_generator:
        print 'Unknown feature'
        print
        sys.exit(1)


    limit = options.limit
    print("token limit: "+str(limit))

    file_encoding = None
    if options.output_format == 'text':
        file_encoding = 'utf-8'

    token_count = 0

    with codecs.open(fn_output, 'wb', file_encoding, buffering=16777216) as out:
        tagset = corpus2.get_named_tagset(options.tagset)
        rdr = corpus2.TokenReader.create_path_reader(options.input_format, tagset, fn_input)

        sentfirst = True  # if non-empty sent sep, skip pre-spaces
        limit_reached = False
        while not limit_reached:
            sent = rdr.get_next_sentence()
            if not sent:
                break

            for tok in sent.tokens():
                if limit and token_count >= limit:
                    limit_reached = True
                    break

                token_count += 1
                feat_val = feature_generator(tok)

                if (sentfirst and options.ignore_ns_sent) or tok.after_space() or options.separate_tokens:
                    out.write(' ')
                out.write(feat_val)
                sentfirst = False

            out.write(options.sent_sep)

    print('token count: '+str(token_count))


if __name__ == '__main__':
    go()
