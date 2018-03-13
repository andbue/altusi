# -*- coding: utf-8 -*-
#
# Requires...

from flask import Flask, render_template, make_response, session, request, abort, redirect, url_for, flash, jsonify, json, g
from werkzeug.serving import run_simple
from werkzeug.wsgi import DispatcherMiddleware

from operator import itemgetter

from purgatorium import *

import xml.etree.ElementTree as ET
import glob, os, sqlite3, shutil, io
from datetime import datetime
#import string, shutil
from tempfile import mkdtemp
from contextlib import closing
#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr
##ro.r('install.packages("stylo")') # nur wenn nicht installiert
#stylo = importr('stylo')

from collections import Counter
import pandas as pd
import collections
from delta import const, Corpus, Delta, Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

########################################################################
#wdir defines the title for stylo-output
wdir = 'Translations_12'
home=os.getcwd()
tempdir = mkdtemp()
SECRET_KEY = 'some_key_for_development'
DATABASE = 'altusi.db'
SOURCESXML = ['J:/Kallimachos/xml/sources.xml', 'Q:/Kallimachos/xml/sources.xml', 'sources.xml']

app = Flask(__name__)
app.config.from_object(__name__)
app.lastrrun=0

########################################################################
#############Functions##################################################

#def create_twdir(texte):
#    print('Temporaeres Verzeichnis: ' + repr(tempdir))
#    twdir = tempdir + '/' + wdir
#    os.mkdir(twdir)
    #os.mkdir(twdir + '\\corpus')
    #write textfiles
    #for i in list(texte.keys()):
    #     with open(twdir + '\\corpus\\'+i+'.txt','w') as fobj:
    #         fobj.write(texte[i])
#    return twdir

#F: lade xml-Datei
def load_xml():
    global xmlroot
    #xml = getattr(g, '_xmltree', None)
    xmlroot = xmlroot if 'xmlroot' in globals() else None
    if xmlroot is None:
        file = [i for i in SOURCESXML if os.path.isfile(i)][0]
        tree = ET.parse(file)
        xmlroot = tree.getroot()
    return xmlroot

#F: transformiere xml in dict
def xmltodict():
    xml = load_xml()
    xdict = {}
    for text in xml.findall("./text/file/.."):
        filename = text.findtext('file')
        xdict[filename] = {}
        for i in ['translator', 'author', 'title', 'shorttitle', 'date-origin', 'source']:
            xdict[filename][i] = text.findtext(i)
        xdict[filename]['number'] = ''.join(n for n in filename.split('_')[1]
            if n.isdigit())
    return xdict

def xmltodict_dates():
    xml = load_xml()
    xdict = xmltodict()
    authors = set([ xdict[x]["author"] for x in xdict.keys() ])
    authors.remove(None)
    authors.remove("Anonymous")
    translators = set([ xdict[x]["translator"] for x in xdict.keys() ])
    translators.remove("Anonymous")
    persondict={}
    for x in xml.findall("./person"):
        name = x.findtext("name")
        if name in (translators | authors):
            persondict[name] = {}
            persondict[name]["gnd"]=x.attrib["gnd"]
            persondict[name]["date"] = int(x.findtext("date").replace("v","-"))
    return persondict

#F: generiere ngram-liste aus liste
def ngram(seq, n):
    if n == 1:
        return seq
    nseq = []
    for i in range(0,len(seq)-n+1):
        wort = seq[i]
        for j in range(i+1,i+n):
            wort += ' ' + seq[j]
        nseq.append(wort)
    return nseq

def tcomp(t1, t2):
    """compare textstrings t1, t2 recursively - return list of 3-tuples ("only_t1", "common", "only_t2")"""
    t1, t2 = t1.strip(), t2.strip()
    if not (len(t1) and len(t2)):
        return [(t1, "", t2)]
    if t1 == t2:
        return [("", t1, "")]
    common = set("")
    t1, t2 = (" " + t + " " for t in (t1, t2) )
    for n in range(1, min(len(t1), len(t2))-1):
        ncommon = set(ngram(t1.split(), n)) & set(ngram(t2.split(), n))
        if not ncommon:
            if not common:
                return [(t1.strip(), "", t2.strip())]
            break
        common = ncommon
    split = " " + common.pop() + " "
    splix = ( t1.index(split), t2.index(split) )
    r1 = ( t1[: splix[0]], t1[splix[0]+len(split) :] )
    r2 = ( t2[: splix[1]], t2[splix[1]+len(split) :] )
    return tcomp(r1[0], r2[0]) + [("",split.strip(),"")] + tcomp(r1[1], r2[1])


#F: lade Textdateien, vergleiche mit Datenbank
def init_filestodb():
    texte={}
    for datei in glob.glob('txts/*.txt'):
        try:
            textfile = open(datei)
        except IOError:
            print('Hilfe, konnte die Datei nicht Ã¶ffnen...')
        else:
            wl = textfile.read()
            textfile.close()
            texte[os.path.splitext(os.path.split(datei)[1])[0]] = wl
    namen=list(texte.keys())
    namen.sort()
    
    alteredfiles=[]
    with app.app_context():
        db = get_db()
        for name in namen:
            db_name = query_db('select * from texts where filename = ?', [name], one=True)
            if db_name is None:
                alteredfiles.append(name)
                db.execute('insert into texts (filename, content) values (?, ?)', [name, texte[name]])
            elif db_name['content'] != texte[name]:
                alteredfiles.append(name)
                db.execute('update texts set content = ? where filename = ?', [texte[name], name])
        print('New files: '+repr(alteredfiles))
        db_result = query_db('select texts from results')
        for i in db_result:
            dbtexts=i['texts']
            for k in alteredfiles:
                if k in dbtexts:
                    db.execute('delete from results where texts = ?', [dbtexts])
        db.commit()
    
    #Replace all whitespace with " "
    for key in texte.keys():
        texte[key]=" ".join(texte[key].split())
    
    return(texte, namen)

#F: Erstelle R-Sylo-datei
def rstellen(indices, settings):
    delta = settings['delta']
    mfwmin, mfwmax, mfwinc = settings['mfwmin'], settings['mfwmax'], settings['mfwinc']
    cullmin, cullmax, cullinc = settings['culmin'], settings['culmax'], settings['culinc']
    ngram, cons = settings['ngram'], settings['cons']
    settings = delta + 'MM'+mfwmin + 'MX'+mfwmax + 'MI'+mfwinc + 'CM'+cullmin + 'CX'+cullmax + 'CI'+cullinc + 'N'+ngram + 'C'+cons
    sel_liste = [ app.namen[x] for x in indices ]
    sel_liste.sort()
    #ident = repr(sel_liste) +  settings
    db_result = query_db('select * from results where texts = ? and settings = ?', [repr(sel_liste), settings], one=True)
    if db_result is not None:
        svg = db_result['svg']
    else:
        ro.r('setwd("'+twdir.replace('\\', '/')+'")')
        lastrrun = app.lastrrun
        
        
        styloargs = {'gui':0, 'display.on.screen':0, 'write.svg.file':1,
                          'corpus.lang':"Latin.corr",'analysis.type':"BCT",
                          'distance.measure':delta,
                          'consensus.strength':float(cons),
                          'culling.incr':int(cullinc), 'culling.max':int(cullmax),
                          'culling.min':int(cullmin),
                          'mfw.incr':int(mfwinc), 'mfw.max':int(mfwmax), 'mfw.min':int(mfwmin),
                          'ngram.size':int(ngram) }
                              
        if lastrrun and indices == lastrrun[0]:
            print('Using existing frequencies...')
            freqtable = lastrrun[1]
            styloargs['frequencies']=freqtable
        else:
            print('Corpus changed...')
            rcorpus=ro.ListVector({})
            for i in sel_liste:
                rcorpus += ro.ListVector( { i : ro.StrVector(app.texte[i].split()) } )
            styloargs['parsed.corpus']=rcorpus
                
        res = stylo.stylo(**styloargs)
        freqtable = res.rx('table.with.all.freqs')[0]
        app.lastrrun=[indices, freqtable]
        os.chdir(home)
        
        imagefile = glob.glob(twdir+'\\*.svg')[0]
        print("imagefile: " + imagefile)
        svg = open(imagefile, encoding="utf8").read()
        os.remove(imagefile)
        db = get_db()
        db.execute('insert into results (texts, settings, svg) values (?, ?, ?)', [repr(sel_liste), settings, svg])
        db.commit()
    return(svg)

def pystellen(indices, settings, format="svg"):
    delta = const._asdict()[ settings['delta'].strip() ]
    mfwmin = int(settings['mfwmin'])
    cullmin = int(settings['culmin'])
    sel_liste = [ app.namen[x] for x in indices ]
    sel_liste.sort()
    #corpus=Corpus(subdir='corpus')
    list_of_wordlists = []
    for text in sel_liste:
        all_words = collections.defaultdict(int)
        #print("processing " + text)
        for w in app.texte[text].split():
            all_words[w] += 1
        wordlist = pd.Series(all_words, name=text)
        list_of_wordlists.append(wordlist)
    
    df = pd.DataFrame(list_of_wordlists).fillna(0).T
    df = df.ix[(-df.sum(axis=1)).argsort()]
    corpus = Corpus(corpus=df)
    
    cul = 1/cullmin    
    corpus = corpus.cull(ratio=cul)
    mfw_corpus = corpus.get_mfw_table(mfwmin)
    deltas = Delta(mfw_corpus, delta, None)
    fig = Figure(deltas.get_linkage('ward'), deltas.index, 'left', 8, 'Latin translations', mfwmin, delta, False)
    #svg = fig.gensvg()
    dendro_dat, plot= fig.show()
    #TEST!!!
    image = fig.genimage(plot, format=format)
    session["lastplot"] = (indices, settings)
    #svg = repr(settings)
    return(image)



def gen_boxplot(indices, settings):
    anonym = 'unbekannt'
    delta = const._asdict()[ settings['delta'].strip() ]
    testtext = settings['testtext'].strip()
    mfwmin = int(settings['mfwmin'])
    cullmin = int(settings['culmin'])
    sel_liste = [ app.namen[x] for x in indices ]
    sel_liste.sort()
    
    if testtext in sel_liste:
        return('Text already in selection!')
    
    sel_liste.append(testtext)
    #corpus=Corpus(subdir='corpus')
    list_of_wordlists = []
    for text in sel_liste:
        all_words = collections.defaultdict(int)
        #print("processing " + text)
        for w in app.texte[text].split():
            all_words[w] += 1
        if text == testtext:
            wordlist = pd.Series(all_words, name=anonym + '_unbekannt')
        else:
            wordlist = pd.Series(all_words, name=text)
        list_of_wordlists.append(wordlist)
    
    df = pd.DataFrame(list_of_wordlists).fillna(0).T
    df = df.ix[(-df.sum(axis=1)).argsort()]
    corpus = Corpus(corpus=df)
    cul = 1/cullmin    
    corpus = corpus.cull(ratio=cul)
    mfw_corpus = corpus.get_mfw_table(mfwmin)
    deltas = Delta(mfw_corpus, delta, None)
    
    authors = list(set([a.split("_")[0] for a in deltas.columns if not a.split("_")[0] == anonym]))
    ingroup_deltas = []
    for i in range(0, len(authors)):
        #alle Werke eines Autors ermitteln
        works = [c for c in deltas.columns if authors[i] in c]
        #alle ingroup-deltas ermitteln und Nullen entfernen
        values = list(set(deltas.loc[works, works].values.flatten()))
        values = [val for val in values if val != 0.0]
        ingroup_deltas.append(values)

    #unbekannten Text zu Ingroup-Werten dazunehmen
    ingroup_deltas_anonym = []
    for i in range(0, len(authors)):
        works = [c for c in deltas.columns if authors[i] in c or anonym in c]
        values = list(set(deltas.loc[works, works].values.flatten()))
        values = [val for val in values if val != 0.0]
        ingroup_deltas_anonym.append(values)

#    #Positionen fÃ¼r die Boxen
    plt.clf()
    pos = list(range(1, len(authors)+1))
    params = ["boxes", "medians", "whiskers", "caps", "fliers", "means"]
    box1 = plt.boxplot(ingroup_deltas, widths=0.3, positions=[p-0.2 for p in pos])
    box2 = plt.boxplot(ingroup_deltas_anonym, widths=0.3, positions=[p+0.2 for p in pos])
#    #Farben fÃ¼r die Boxen
    for p in params:
        plt.setp(box1[p], color="blue")
        plt.setp(box2[p], color="red")
#    #Achseneigenschaften
    plt.xticks(np.arange(1, len(authors)+1), authors)
    plt.xlim(xmin=0.5)   
    
    output = io.StringIO()
    plt.savefig(output, format='svg')
    svg = output.getvalue()
    output.close()     
    
    #fig = Figure(deltas.get_linkage('ward'), deltas.index, 'left', 8, 'Latin translations', mfwmin, delta, False)
    #svg = fig.gensvg()
    #dendro_dat, plot, svg= fig.show()
    #svg = repr(settings)
    return(svg)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

########DB-Helpers#####################################################

def connect_db():
    return sqlite3.connect(app.config['DATABASE'])

def init_db():
    with closing(connect_db()) as db:
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv
    
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_db()
    db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
        
########################################################################




###################### FLASK ###########################################


@app.route('/', methods=['POST', 'GET'])
def start():
    #cur = get_db().cursor()
    #content = app.namen
    #date = datetime.datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    #return render_template('basis.html', content=content, date=date)
    return redirect(url_for('pydelta'))

#@app.route('/rstylo', methods=['POST', 'GET'])
#def rstylo():
#    #cur = get_db().cursor()
#    content = []
#    textinfo = xmltodict()
#    for i in app.namen:
#        content.append([i, textinfo[i]['title']])
#    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
#    return render_template('rstylo.html', content=content, date=date)
    
    
@app.route('/pydelta', methods=['POST', 'GET'])
def pydelta():
    #cur = get_db().cursor()
    content = []
    textinfo = xmltodict()
    deltas = list(const._fields)
    for i in app.namen:
        content.append([i, textinfo[i]['title']])
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('pydelta.html', content=content, deltas=deltas, date=date)

@app.route('/boxplot', methods=['POST', 'GET'])
def boxplot():
    #cur = get_db().cursor()
    content = []
    textinfo = xmltodict()
    deltas = list(const._fields)
    for i in app.namen:
        content.append([i, textinfo[i]['title']])
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('boxplot.html', content=content, deltas=deltas, date=date)

@app.route('/_select_texts', methods = ['POST'])
def select_texts():
    data = request.json
    indices = [ int(x) for x in data['indices'].keys() ]
    #print(repr(data['settings']))
    settings=(data['settings'])
    if len(indices) < 3:
        svg = 'Choose at least three texts!'
    else:
        svg=rstellen(indices, settings)
    return jsonify(svgimage=svg)

@app.route('/_select_pydelta', methods = ['POST'])
def select_pydelta():
    data = request.json
    indices = [ int(x) for x in data['indices'].keys() ]
    #print(repr(data['settings']))
    settings=(data['settings'])
    if len(indices) < 3:
        svg = 'Choose at least two texts!'
    else:
        #svg = repr(settings)
        svg=pystellen(indices, settings)
    return jsonify(svgimage=svg)

@app.route('/_select_boxplot', methods = ['POST'])
def select_boxplot():
    data = request.json
    indices = [ int(x) for x in data['indices'].keys() ]
    #print(repr(data['settings']))
    settings=(data['settings'])
    if len(indices) < 3:
        svg = 'Choose at least two texts!'
    else:
        #svg = repr(settings)
        svg=gen_boxplot(indices, settings)
    return jsonify(svgimage=svg)

@app.route('/_searchbasic', methods = ['POST','GET'])
def searchbasic():
    #data = request.json
    textinfo=xmltodict()
    data = request.json
    searchterm = data['settings']['sterm']
    completew = data['settings']['completew']
    sortdate = True if data['settings']['sortdate'] == "True" else False
    indices = [ int(x) for x in data['indices'].keys() ]
    sel_liste = [ app.namen[x] for x in indices ]
    new_keys = [x for x in app.texte.keys() if x not in sel_liste]
    compl = True if completew == "True" else False
    result=[]
    rowsp = 0
    auth=""
    new_keys.sort()
    
    if sortdate:
        texttodate = []
        authorsdict = xmltodict_dates()
        for k in new_keys:
            author = textinfo[k]["author"]
            translator = textinfo[k]["translator"]
            if author not in authorsdict.keys():
                if textinfo[k]["date-origin"]:
                    adate = int(textinfo[k]["date-origin"].replace("v","-"))
                else:
                    adate = -10000
            else:
                adate = authorsdict[author]["date"]
            if translator not in authorsdict.keys():
                tdate = -10000
            else:
                tdate = authorsdict[translator]["date"]
            texttodate.append([k, tdate, adate])
        texttodate = sorted(texttodate, key=itemgetter(1,2,0))
        new_keys = [x[0] for x in texttodate]
    
    for i in new_keys:
        tn=i.split("_",1)
        if (tn[0] == auth):
            result[-rowsp][4] += 1
            rowsp += 1
            span = 0
        else:
            rowsp, span = 1,1
        
        if compl:
            result += [ [tn[0], tn[1], ngram(app.texte[i].split(), len(searchterm.split())).count(searchterm.lower()), i, span]  ]
        else:
            result += [ [tn[0], tn[1], app.texte[i].count(searchterm.lower()), i, span]  ]

        auth = tn[0]
        
    for k in new_keys:
        textinfo[k]["len"] = len(app.texte[k].split())

    sp="\t"
    csvtable=""
    maxlen = max([ textinfo[k]["len"] for k in new_keys])

    for r in result:
        shortt = textinfo[r[3]]["shorttitle"]
        if not shortt:
            auth = textinfo[r[3]]["author"]
            title = textinfo[r[3]]["title"]
            if not auth:
                auth = "Anonymous"
            shortt = auth + ": " + title
        rellen = "({0:.2f}%)".format( textinfo[r[3]]["len"] / maxlen * 100 )
        nr = textinfo[r[3]]["number"]

        csvtable+=r[0]+sp+nr+sp+shortt+sp+rellen+sp+repr(r[2])+"\r\n"
    #result.sort()
    result=render_template('textsearch_results.html', content=result, csvtable=csvtable, sterm = searchterm)
    return jsonify(result=result)

@app.route('/_genwl', methods = ['POST','GET'])
def genwl():
    data = request.json
    trh = int(data['settings']['trh'])
    ngraml = int(data['settings']['ngram'])
    maxfehler = int(data['settings']['mxerror'])
    translator = data['settings']['translator']
    cull = int(data['settings']['cull'])    
    indices = [ int(x) for x in data['indices'].keys() ]
    sel_liste = [ app.namen[x] for x in indices ]
    new_keys = [x for x in app.texte.keys() if x not in sel_liste]
    new_keys.sort()
    autortext=[]
    andretext=[]
    ntexte={}
    for text in new_keys:
        ntexte[text] = ngram(app.texte[text].split(), ngraml)
        if translator == text.split('_',2)[0]:
            autortext += ntexte[text]
        elif 'ANON' != text.split('_',2)[0]:
            andretext += ntexte[text]
    andrecount = Counter( andretext )
    counts = Counter( autortext ).most_common()
    eigenw = []
    for wcnt in counts:
        if wcnt[1] > trh and andrecount[ wcnt[0] ] <= maxfehler :
            eigenw.append(wcnt)
            
    if cull > 0:
        autorkeys = [x for x in new_keys if x.split('_',2)[0].startswith(translator)]
        req = int(cull/100 * len(autorkeys))
        ausgangspunkt = eigenw[:]
        for i in ausgangspunkt:
            cnt=0
            for text in autorkeys:
                if i[0] in ntexte[text]:
                    cnt += 1
            if cnt < req:
                eigenw.remove(i)
    
    wlist = []
    for i in eigenw:
        anontxts = ""
        for key in [ x for x in new_keys if (x.split('_',1)[0]=='ANON') ]:
            cnt = ntexte[key].count(i[0])
            if cnt:
                anontxts += (key.split('_',1)[1] + "("+repr(cnt)+"); ")
        wlist.append([ i[0], i[1], anontxts ])
    
    result = render_template('wordlist_results.html', content = wlist)

    return jsonify(result=result)

@app.route('/_comparetexts', methods = ['POST','GET'])
def comparetexts():
    data = request.json
    right = data["texts"]["right"].split()
    left = data["texts"]["left"].split()    
    result = repr(right) + repr(left)
    result = "works as expected: " + result
    return jsonify(result=result)

@app.route('/_concordance', methods = ['POST','GET'])
def concordance():
    data = request.json
    sterm = data['settings']['sterm']
    text = data['settings']['text']
    completew = data['settings']['completew']
    compl = True if completew == "True" else False
    output=[]
    if compl:
        textdata = app.texte[text].lower().split()    
        sterms = sterm.split()
        env = 10
        div = " "
    else:
        textdata = app.texte[text].lower()
        sterms = sterm
        env = 50
        div = ""
    ctlen=len(textdata) + 1 - len(sterms)
    for k in range(ctlen):
        if sterms == textdata[k:(k+len(sterms))]:
            if k < env:
                start = 0
            else:
                start=k-env
            if k > (ctlen - len(sterms) - env):
                stop = ctlen
            else:
                stop=k+env+1
            vorher=''
            danach=''
            for pos in range(start,k):
                vorher+= textdata[pos] + div
            for pos in range(k+len(sterms),stop):
                danach+= div + textdata[pos]
            output.append([vorher, sterm, danach])    
    
                
    result = render_template('concordance_results.html', content = output)
    return jsonify(result=result)

@app.route('/_bookmarks', methods = ['POST', 'GET'])
def bookmarks():
    data = request.json
    addbm = data['settings']['addbm'].strip()
    indices = [ int(x) for x in data['indices'].keys() ]
    sel_liste = [ app.namen[x] for x in indices ]
    sel_liste.sort()
    db_bm = query_db('select * from bookmarks')
    bookmarks = {}
    for i in db_bm:
        if i["name"] != addbm:
            bookmarks[i["name"]] = [ app.namen.index(x) for x in json.loads( i["texts"] ) 
                    if x in app.namen]
        else:
            db=get_db()
            if len(indices):
                db.execute('update bookmarks set texts = ? where name = ?', [json.dumps(sel_liste), addbm])
                bookmarks[addbm] = indices
            else:
                db.execute('delete from bookmarks where name = ?', [addbm])
            db.commit()            
            #db_bm["settings"]

    if addbm and indices and addbm not in [ x["name"] for x in db_bm ]:
        bookmarks[addbm] = indices
        db=get_db()
        db.execute('insert into bookmarks (name, texts) values (?, ?)', [addbm, json.dumps(sel_liste)])
        db.commit()

    result = render_template( 'bookmarks.html', content = bookmarks.keys() )        
    return jsonify(content=result, bmtable=bookmarks)


@app.route('/textsearch', methods=['POST', 'GET'])
def textsearch():
    content = []
    textinfo = xmltodict()
    for i in app.namen:
        content.append([i, textinfo[i]['title']])
    #cur = get_db().cursor()
    authors = list(set( [ x.split('_',2)[0] for x in app.namen ] ))
    authors.remove('ANON')
    authors.sort()
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('textsearch.html', content=content, authors=authors, texts=app.namen, date=date)

@app.route('/compare', methods=['POST', 'GET'])
def compare():
    #cur = get_db().cursor()
    content = []
    textinfo = xmltodict()
    for i in app.namen:
        content.append([i, textinfo[i]['title']])
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('compare.html', content=content, texts=app.namen, date=date)

@app.route('/synopsis')
def synopsis():
    #cur = get_db().cursor()
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('synopsis.html', content=app.namen, date=date)

@app.route('/_gensynopsis', methods=['POST', 'GET'])
def gensynopsis():
    data = request.json
    #print(repr(data['settings']))
    settings=data['settings']
    t1 = settings['text1'].strip()
    t2 = settings['text2'].strip()
    h1, h2 = (t+" ("+repr(len(app.texte[t].split()))+"W)" for t in (t1, t2))
    head = [[h1, 0, h2]]
    content = tcomp(app.texte[t1], app.texte[t2])
    return jsonify(result = render_template('synopsis_results.html', head=head, content=content))
    
@app.route('/sources', methods=['POST', 'GET'])
def sources():
    #cur = get_db().cursor()
    content=[]
    xml = load_xml()
    for text in xml.findall("./text/file/.."):
        entry=[text.findtext('file'), text.findtext('translator'), text.findtext('title'), text.findtext('author'), text.findtext('source')]
        entry.append(repr(len(app.texte[entry[0]].split())))
        content.append(entry)
    content = sorted(content, key=lambda x: x[0])
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('sources.html', content=content, date=date)

@app.route('/authors', methods=['POST', 'GET'])
def authors():
    #cur = get_db().cursor()
    content=[]
    authorsdict = xmltodict_dates()
    for a in authorsdict:
        entry = [ a, str(authorsdict[a]['date']), authorsdict[a]["gnd"] ]
        content.append(entry)
    #content = sorted(content, key=lambda x: x[0])
    content = sorted(content, key=lambda x: int(x[1]))
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('authors.html', content=content, date=date)
    
@app.route('/sourcesreload')
def sourcesreload():
    global xmlroot
    xmlroot = None
    return redirect(url_for('sources'))
    
@app.route('/textdownload', methods=['POST', 'GET'])
def textdownload():
    textname = request.args.get('textname', 'empty', type=str)
    text='Dies ist ein Test'
    if textname != 'empty':
        text = app.texte[textname]
    response = make_response(text)
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = "attachment; filename=" + textname + ".txt"
    return response

@app.route('/sourcesdl')
def sourcesdl():
    with open('sources.xml', 'rb') as f:
        xml = f.read()
    response = make_response(xml)
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/xml'
    response.headers['Content-Disposition'] = "attachment; filename=sources.xml"
    return response

@app.route('/imgdownload', methods=['POST', 'GET'])
def imgdownload():
    imgtype = request.args.get('imgtype', 'empty', type=str)
    if imgtype not in ('svg', 'png'):
        imgtype = 'svg'
        
    lastplot = session.get('lastplot', None)
    if not lastplot:
        return ""
    image = pystellen(lastplot[0], lastplot[1], imgtype)   
    
    response = make_response(image)
    
    mimetype = 'image/svg+xml' if imgtype == 'svg' else 'image/' + imgtype
    
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = mimetype
    response.headers['Content-Disposition'] = "attachment; filename=altusi_plot." + imgtype
    return response

@app.route('/txtinfo')
def textinfo():
    retstring=""
    for t in sorted(app.purgatorium.rawcorpus.texts):
        retstring += t
    
    return retstring

@app.route('/txtedit', methods=['POST', 'GET'])
def txtedit():
    textname = request.args.get('textname', 'empty', type=str)
    page = request.args.get('page', 1, type=int)
    pagesplit = 20000
    if textname not in app.purgatorium.rawcorpus.texts:
        textname = sorted(app.purgatorium.rawcorpus.texts)[0]
    if textname != "empty" and textname in app.purgatorium.rawcorpus.texts:
        text = [ x for x in enumerate(app.purgatorium.lemmatext(textname))]
    nrpages = (len(text) // pagesplit) + 1
    textpage = text[((page-1)*pagesplit):(page*pagesplit)]
    page_p = page - 1 if page>1 else nrpages
    page_n = page + 1 if page<nrpages else 1
    pageswitch = [ textname, repr(page_p), repr(page), repr(nrpages), repr(page_n) ]
    content = [(x, "") for x in sorted(app.purgatorium.rawcorpus.texts)]
    date = datetime.now().strftime('%B %d, %Y - %H:%M (local)')
    return render_template('txtedit.html', text=textpage, pageswitch=pageswitch, content=content, date=date)

@app.route('/_saveedits', methods=['POST', 'GET'])
def saveedits():
    data = request.json
    edits = data["edits"]
    textname = data["params"]["textname"]
    text = app.purgatorium.textsplit(app.purgatorium.rawcorpus.texts[textname])
    changes=[]
    for i in edits:
        eid = int(i)
        old = text[eid]
        new = edits[i]
        repl=[("\xa0", " ")]
        for r in repl:
            new = new.replace(r[0], r[1])
        if old != new:
            changes.append( (eid, old, new) )
    for c in changes:
        text[c[0]] = c[2]
    if changes:
        newtext = "".join(text)
        newtext_splt = app.purgatorium.textsplit(newtext)
        newtext_wlist = set(newtext_splt[1::2])
        app.purgatorium.wwordslemmacache = app.purgatorium.gen_whitakerdict(newtext_wlist, app.purgatorium.wwordslemmacache)
        app.purgatorium.rawcorpus.texts[textname] = newtext
        app.purgatorium.picklecompr(app.purgatorium.corpusfile, app.purgatorium.rawcorpus)
    with open("debug.txt", "a") as f:
        f.write(repr(changes))
    return 'Changes saved...'

@app.route('/shutdown', methods=['POST', 'GET'])
def shutdown():
    #ro.rinterface.endr(0)
    print('Delete Tempdir')
    os.chdir(home)
    #shutil.rmtree(tempdir)
    shutdown_server()
    return 'Server shutting down...'

#@app.before_request
#def limit_remote_addr():
#    if '132.187.111' not in request.remote_addr:
#        print(request.remote_addr)
#        abort(403)  # Forbidden

@app.route('/study', methods=['POST', 'GET'])
def study():
    if request.method == 'POST':
        req = request.form['search']
        flash('Request: ' + repr(req))
    return redirect(url_for('start'))


def simple(env, resp):
    resp(b'200 OK', [(b'Content-Type', b'text/plain')])
    return [b'altusi - redir /altusi']

app.texte, app.namen = init_filestodb()
app.config['APPLICATION_ROOT'] = '/altusi'

wwords = Wwords()
app.purgatorium = Purgatorium(corpus="rawcorpus.pklz", wwords=wwords, wwordslemmacache="whdict.pklz")

parent_app = DispatcherMiddleware(simple, {'/altusi':app})

if __name__ == '__main__':
    run_simple('localhost', 5000, parent_app)
#    twdir = create_twdir(texte)
#    app.run(debug=True)
    #app.run(host='0.0.0.0', debug=True, port=5050) # public server
