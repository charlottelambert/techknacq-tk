# TechKnAcq: Corpus
# Jonathan Gordon

import sys
import os
import io
import json
import datetime
import re
import multiprocessing as mp
import ftfy

from pathlib import Path
from bs4 import BeautifulSoup
#from xml.sax.saxutils import escape
#from unidecode import unidecode
from nltk import bigrams

from techknacq.lx import SentTokenizer, StopLexicon, find_short_long_pairs

class Corpus:
    def __init__(self, path=None, pool=None, path_list=None):
        self.docs = {}

        if path and os.path.isfile(path):
            print("Reading data from", path)
            if ".tsv" in path:
                with open(path, 'r') as f:
                    docs = f.read().split("\n")
                    for d in docs[1:]:
                        doc = Document(text=d, form='tsv')
                        try:
                            self.add(doc)
                        except:
                            print("Error with this line:", d)
                            continue
            else:
                # Read a BioC corpus file.
                j = json.load(open(path))
                for d in j['documents']:
                    doc = Document()
                    doc.read_bioc_json(d)
                    self.add(doc)
        elif path:
            if not pool:
                pool = mp.Pool(int(.5 * mp.cpu_count()))
            if path_list:
                docnames = tuple(path_list)
            else:
                docnames = (str(f) for f in Path(path).iterdir() if f.is_file() and str(f)[-4:] == ".txt" and re.match(".*[0-9]{8}", str(f)))
            for doc in pool.imap(Document, docnames):
                if doc:
                    self.add(doc)
            print('Read %d documents.' % len(self.docs))

    def clear(self):
        self.docs = {}

    def add(self, doc):
        assert(type(doc) == Document)
        doc.corpus = self
        self.docs[doc.id] = doc

    def __ior__(self, other):
        for doc in other:
            self.add(doc)
        return self

    def __iter__(self):
        for doc_id in self.docs:
            yield self.docs[doc_id]

    def __getitem__(self, key):
        return self.docs[key]

    def __setitem__(self, key, item):
        self.docs[key] = item

    def __contains__(self, key):
        return key in self.docs

    def fix_text(self):
        pass
        #for doc in self:
            #doc.dehyphenate()
            #doc.expand_short_forms()

    def export(self, dest, abstract=False, form='json'):
        if form not in ['json', 'bioc', 'text', 'bigrams', 'tsv']:
            print('Unrecognized form for export', form, file=sys.stderr)
            sys.exit(1)

        if form == 'bigrams':
            stop = StopLexicon()

        for d in self:
            if form == 'json':
                with io.open(os.path.join(dest, d.id + '.json'), 'w',
                             encoding='utf-8') as out:
                    out.write(d.json(abstract) + '\n')
            elif form == 'bioc':
                with io.open(os.path.join(dest, d.id + '.xml'), 'w',
                             encoding='utf-8') as out:
                    out.write(d.bioc(abstract) + '\n')
            elif form == 'text':
                with io.open(os.path.join(dest, d.id + '.txt'), 'w',
                             encoding='utf-8') as out:
                    out.write(d.text(abstract) + '\n')
            elif form == 'bigrams':
                with io.open(os.path.join(dest, d.id + '.txt'), 'w',
                             encoding='utf-8') as out:
                    out.write(d.bigrams(abstract, stop) + '\n')
            elif form == 'tsv':
                with io.open(os.path.join(dest, d.id + '.tsv'), 'w',
                             encoding='utf-8') as out:
                    out.write(d.text(abstract) + '\n')


class Document:
    def __init__(self, fname=None, form=None, text=""):
        if fname and not form:
            if 'json' in fname:
                form = 'json'
            elif 'xml' in fname:
                form = 'sd'
            elif 'txt' in fname:
                form = 'text'

        j = {'info': {}}
        if fname and form == 'json':
            try:
                j = json.load(io.open(fname, 'r', encoding='utf-8'))
            except Exception as e:
                print('Error reading JSON document:', fname, file=sys.stderr)
                print(e, file=sys.stderr)
                sys.exit(1)

        if 'id' in j['info']:
            self.id = j['info']['id']
        elif fname:
            basename = os.path.basename(fname)
            basename = re.sub('\.(json|xml|txt)$', '', basename)
            self.id = basename
        self.authors = [x.strip() for x in j['info'].get('authors', [])]
        self.title = title_case(j['info'].get('title', ''))
        self.book = title_case(j['info'].get('book', ''))
        self.year = j['info'].get('year', '')
        self.url = j['info'].get('url', '')
        self.references = set(j.get('references', []))
        self.sections = j.get('sections', [])
        self.roles = {}
        self.corpus = None

        if fname and form == 'text':
            st = SentTokenizer()
            text = open(fname).read().split()
            text = " ".join([word for word in text if not word[0].isdigit()])
            content = st.tokenize(text)
            self.sections = [{'text': content}]
        elif text and form == 'tsv':
            st = SentTokenizer()
            id, year, t = text.split("\t")
            sents = st.tokenize(t)
            sents
            self.sections = [{'text': sents}]
            self.year = year
            self.id = id

    def dehyphenate(self):
        """Fix words that were split with hyphens."""

        def dehyphenate_sent(s):
            words = s.split()
            out = []
            skip = False
            for w1, w2 in bigrams(words):
                if skip:
                    skip = False
                elif w1[-1] == '-':
                    if d.check(w1[:-1] + w2):
                        out.append(w1[:-1] + w2)
                        skip = True
                    elif w1[0].isalpha() and w2 != 'and':
                        out.append(w1 + w2)
                        skip = True
                    else:
                        out.append(w1)
                else:
                    out.append(w1)
            if not skip:
                out.append(words[-1])
            return ' '.join(out)

    def get_abstract(self):
        """Return the (probable) abstract for the document."""

        if self.sections[0].get('heading', '') == 'Abstract':
            return self.sections[0]['text'][:10]
        if len(self.sections) > 1 and \
           self.sections[1].get('heading', '') == 'Abstract':
            return self.sections[1]['text'][:10]

        if len(self.sections[0]['text']) > 2:
            return self.sections[0]['text'][:10]
        if len(self.sections) > 1 and len(self.sections[1]['text']) > 2:
            return self.sections[1]['text'][:10]
        return self.sections[0]['text'][:10]


    def json(self, abstract=False):
        """Return a JSON string representing the document."""

        doc = {
            'info': {
                'id': self.id,
                'authors': self.authors,
                'title': self.title,
                'year': self.year,
                'book': self.book,
                'url': self.url
            },
            'references': sorted(list(self.references))
        }
        if abstract:
            doc['sections'] = [self.sections[0]]
        else:
            doc['sections'] = self.sections

        return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False)

    def text(self, abstract=False):
        """Return a plain-text string representing the document."""
        out = self.title + '.\n'
        out += self.title + '.\n'

        for author in self.authors:
            if author == 'Wikipedia':
                continue
            out += author + '.\n'

        out += self.book + '\n' + str(self.year) + '\n\n'

        if abstract:
            out += '\n'.join(self.get_abstract())
        else:
            for sect in self.sections:
                if 'heading' in sect and sect['heading']:
                    out += '\n\n' + sect['heading'] + '\n\n'
                out += '\n'.join(sect['text']) + '\n'

        for ref_id in sorted(list(self.references)):
            if not ref_id in self.corpus:
                continue
            out += '\n'
            for author in self.corpus[ref_id].authors:
                if author == 'Wikipedia':
                    continue
                out += author + '.\n'
            out += self.corpus[ref_id].title + '.\n'
        return out
        #return filter_non_printable(unidecode(out))


    def bigrams(self, abstract=False, stop=StopLexicon()):
        def good_word(w):
            if '_' in w and not '#' in w:
                return False
            return any(c.isalpha() for c in w)

        def bigrams_from_sent(s):
            #s = unidecode(s)
            ret = ''
            words = []
            for x in re.split(r'[^a-zA-Z0-9_#-]+', s):
                if len(x) > 0 and not x in stop and not x.lower() in stop \
                   and re.search('[a-zA-Z0-9]', x) and not x[0].isdigit():
                    words.append(x)
            for w1, w2 in bigrams(words):
                if good_word(w1) and good_word(w2):
                    ret += w1.lower() + '_' + w2.lower() + '\n'
                if w1[0] == '#' and w1[-1] == '#' and good_word(w1):
                    ret += w1 + '\n'
            if words and words[-1][0] == '#' and words[-1][-1] == '#' and \
               good_word(words[-1]):
                ret += words[-1] + '\n'
            return ret

        out = bigrams_from_sent(self.title)
        out += bigrams_from_sent(self.title)

        for author in self.authors:
            if author == 'Wikipedia':
                continue
            out += bigrams_from_sent(author)

        out += bigrams_from_sent(self.book)

        if abstract:
            for sent in self.get_abstract():
                out += bigrams_from_sent(sent)
        else:
            for sect in self.sections:
                if 'heading' in sect and sect['heading']:
                    out += bigrams_from_sent(sect['heading'])
                for sent in sect['text']:
                    out += bigrams_from_sent(sent)

        for ref_id in sorted(list(self.references)):
            if not ref_id in self.corpus:
                continue
            for author in self.corpus[ref_id].authors:
                if author == 'Wikipedia':
                    continue
                out += bigrams_from_sent(author)
            out += bigrams_from_sent(self.corpus[ref_id].title)

        return out


def filter_non_printable(s):
    return ''.join([c for c in s if ord(c) > 31 or ord(c) == 9 or c == '\n'])

def title_case(s):
    for word in ['And', 'The', 'Of', 'From', 'To', 'In', 'For', 'A', 'An',
                 'On', 'Is', 'As', 'At']:
        s = re.sub('([A-Za-z]) ' + word + ' ', r'\1 ' + word.lower() + ' ', s)
    return s

def strtr(text, dic):
    """Replace the keys of dic with values of dic in text."""
    pat = '(%s)' % '|'.join(map(re.escape, dic.keys()))
    return re.sub(pat, lambda m:dic[m.group()], text)
