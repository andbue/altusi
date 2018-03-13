from lxml import etree
import string, sys, os, glob, ctypes, subprocess, threading, re, itertools, pickle, gzip
from queue import Queue

#from collections import defaultdict



def morpheus(word):
    """Ask morpheus from Perseus for a word. Has to be in folders "./M" and "./M/stemlib"."""
    inp = bytes(word+'\n', 'utf-8')
    env = os.environ.copy()
    env['MORPHLIB'] = 'M/stemlib'
    proc = subprocess.Popen(['M/cruncher', '-Ll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, shell=False)
    grep_stdout = proc.communicate(input=inp)[0]
    return {word : grep_stdout.decode().split()[1:]}


class Wwords:
    """
    Access to Whitaker's Words (http://mk270.github.io/whitakers-words), xml-enabled by Alpheios:
    http://sourceforge.net/p/alpheios/code/HEAD/tarball?path=/wordsxml/trunk
    Build some ADA-Wrappers around the PARSE-Function, gnatmake -O3 -Ppyparse.
    Function "ask" returns xml-Etree of Words reply.
    Maximum input is limited to 2500 chars - maybe use another pipe for input?
    """
    def __init__(self):
        self.ada=ctypes.CDLL("W/libpyparse.so")
        self.ada.initwords()

    def ask(self, latin):
        """Ask Whitaker's Words for latin word, returns entry (xml) as string."""
        if not latin:
            return ""

        #reroute stdout to pipe
        stdout_fileno = sys.__stdout__.fileno()
        stdout_save = os.dup(stdout_fileno)
        stdout_pipe = os.pipe()
        os.dup2(stdout_pipe[1], stdout_fileno)

        #helper function to avoid pipe deadlock (windows...)
        def drain_pipe(q):
            output=""
            while True:
                data = os.read(stdout_pipe[0], 1024)
                output += data.decode("utf-8")
                if '</words>' in output:
                    q.put_nowait(output)
                    break
                
        queue = Queue()
        t = threading.Thread(target=drain_pipe, args=(queue,))
        t.start()
        
        #call ada function
        self.ada.words(bytes(latin, "utf-8"))  # Call into the shared library
        
        t.join() #wait for thread to finish

        # close pipes and reroute stdout
        os.close(stdout_pipe[1])
        os.close(stdout_pipe[0])
        os.close(stdout_fileno)
        os.dup2(stdout_save, stdout_fileno)
        os.close(stdout_save)
                
        return queue.get_nowait()
    
    

class Rawcorpus:
    """
    Baue Corpus und speichere in Datenbank (optional?)
    """
    
    def __init__(self):
        self.texts = {}
        #self.load_texts()

    #def savetodb(self):
    #    print('save')

    #def connect_db(self):
    #    self.db = []

    #def gen_db(self):
    #    self.db = []

    def load_texts(self, sourcexml, sources_folder):
        """
        Load sources.xml and return dict containing filename, translator, title, source.
        """
        tree = etree.parse(sourcexml)
        xml = tree.getroot()
        xdict = {}
        for text in xml.findall("./text/file/.."):
            filename = text.findtext('file')
            xdict[filename] = {}
            xdict[filename] = {}
            xdict[filename]['translator'] = text.findtext('translator')
            xdict[filename]['title'] = text.findtext('title')
            xdict[filename]['source'] = text.findtext('source')
        txt_files = [ glob.glob(sources_folder+i+'/*'+i+'.txt')[0] for i in os.listdir(sources_folder)]
        for tname in xdict.keys():
            print("loading text "+tname)
            file = [i for i in txt_files if tname in i][0]
            with open(file, encoding="utf8", errors="ignore", mode="r") as f:
                self.texts[tname] = f.read()
            #Text(file=file, name=tname, descr=xdict[tname])


class Purgatorium:
    """
    Cleans texts. corpus may be rawcorpus or filename, wwordslemmacache may be dict or filename.
    Used like that:
    #wwords = Wwords()
    #purgatorium = Purgatorium(corpus="rawcorpus.pklz", wwords=wwords, wwordslemmacache="whdict.pklz")
    """

    def __init__(self, corpus={}, wwords=None, wwordslemmacache={} ):
        self.rawcorpus = corpus
        self.wwords=wwords
        if type(wwordslemmacache) == dict:
            self.lemmafile = "wwordslemmacache-default.pklz"
            self.wwordslemmacache = wwordslemmacache
        if type(wwordslemmacache) == str:
            self.lemmafile = wwordslemmacache
            self.wwordslemmacache = self.pickledecompr(self.lemmafile)

        if type(wwordslemmacache) in (dict, Rawcorpus):
            self.corpusfile = "rawcorpus-dump-default.pklz"
            self.rawcorpus = corpus
        if type(corpus) == str:
            self.corpusfile = corpus
            self.rawcorpus = self.pickledecompr(self.corpusfile)
            
        wordlist = set([ x for y in self.rawcorpus.texts.values() for x in self.textsplit(y)[1::2]])
        self.wwordslemmacache = self.gen_whitakerdict(wordlist, self.wwordslemmacache)
        
        self.picklecompr(self.lemmafile, self.wwordslemmacache)
        #self.morpheus = []
        #self.lexica = []
    
    def clean(self, text=[]):
        """Transform texts, return clean corpus."""
        return
    
    def save(self):
        """Save Purgatorium to file"""
        # to be saved: ???
        return
    
    def load(self):
        """Load Purgatorium from file"""
        return
    
    def load_rules(self, string):
        """Load string containing replacement rules"""
        return
        
    def picklecompr(self, filename, obj):
        with gzip.open(filename, "wb") as f:
            pickle.dump(obj, f)
    
    def pickledecompr(self, filename):
        with gzip.open(filename,'rb') as f:
            return pickle.load(f)
    
    
    def textinfo(self):
        """Show Textinfo (status, errors, etc...)"""
        maxlen = max(len(x) for x in self.rawcorpus.texts)
        for t in sorted(self.rawcorpus.texts):
            text = self.textsplit(self.rawcorpus.texts[t])[1::2]
            qual = len([x for x in text if self.wwordslemmacache[x]])/len(text)
            fstr = "{:<"+repr(maxlen)+"} Q: {:.2%}, L: {:>6}W"
            print(fstr.format(t, qual, len(text)))
            
            
    def textsplit(self, text=""):
        regsplit = re.compile("([^\W\d_]+)")
        splits = regsplit.split(text)
        #splits[1::2] - words, splits[1::2] - punctuation etc.
        return splits
    
    def lemmatext(self, textname):
        splits = self.textsplit(self.rawcorpus.texts[textname])
        lemmata = []
        for i, w in enumerate(splits):
            if not i%2:
                lemmata.append(w)
            else:
                lemmata.append([w, self.wwordslemmacache[w]])
        return lemmata

    def showchars(self, allowed=set(string.ascii_letters)):
        for i in sorted(self.rawcorpus.texts):
            print(i + ":\t"+"".join(sorted(list(set(self.rawcorpus.texts[i])-allowed))))
            
    def showall(self, searchterm, textname=None, around=25, occ=5):
        if not textname:
            texts = self.rawcorpus.texts.keys()
        else:
            texts = [textname]
        pattern = re.compile('[\S\s]{0,'+str(around)+'}'+searchterm+'[\s\S]{0,'+str(around)+'}')
        for t in sorted(texts):
            #res = re.findall('[\S\s]{0,'+str(around)+'}'+searchterm+'[\s\S]{0,'+str(around)+'}', self.rawcorpus.texts[t])
            res = pattern.finditer(self.rawcorpus.texts[t])
            res = [repr(x.group(0)) for x in itertools.islice(res, occ)]
            print("\n".join([t+":"]+res))    
            
    def search(self, searchterm=""):
        for i in sorted(self.rawcorpus.texts):
            print(i+":\t"+repr(self.rawcorpus.texts[i].count(searchterm)))
    
    def gen_whitakerdict(self, wordlist, whdict={}):
        print("_"*100)
        prttrigger = len(wordlist) // 100
        print("|"*(len(whdict)//prttrigger), end="")
        for w in set(wordlist)-set(whdict.keys()):
            try:
                whdict[w] = self.whitakerlemma(w)
            except (AttributeError, KeyError, IndexError):
                print("Error!!!: " + w)
                return whdict
            if not len(whdict)%prttrigger:
                print("-", end="")
        return whdict
        
    def whitakerlemma(self, word):
        xml = self.whitakerxml(word)
        if xml.xpath("unknown"):
            return ""
        if xml.xpath("error"):
            return ""
        if not len(xml):
            return ""
        if any(x.text.startswith("May be 2 words") for x in xml.xpath(".//mean") if type(x.text)==str) or len(xml.xpath("word"))>1:
            return ""
        hdwd = set([e.text.split(",")[0] for e in xml.xpath(".//hdwd")])
        if len(hdwd) > 1: #Take the one with the highest freq rank.
            sortedxml = sorted(xml.xpath(".//dict[hdwd]/freq[@order]/../.."), key=lambda x: x.xpath("dict/freq/@order")[0])
            if not sortedxml:
                lemma = list(hdwd)[0]
            else:
                lemma = sortedxml[-1].xpath("dict/hdwd")[0].text.split(",")[0]        
        elif len(hdwd) == 1:
            lemma = list(hdwd)[0]
        elif len(hdwd) == 0: #Return just the stem.
            lemma = xml.xpath(".//stem")[0].text
        return lemma
    
    def whitakerxml(self, word):
        return etree.fromstring(self.wwords.ask(word))

