# TechKnAcq Toolkit

This is a minimal implementation of the TechKnAcq system for generating
reading lists.


## Components

Tools:
- *build-corpus*:
  Given a directory of PDF or text files, create a directory of JSON files
  containing the document text, annotated with the features needed to judge
  their inclusion in a reading list, and download related encyclopedia
  articles, book chapters, and tutorials.
- *concept-graph*:
  Analyze a JSON corpus to return a JSON graph of concepts and documents
  with the features and links needed to find documents for a reading
  list.
- *reading-list*:
  Given a concept graph, return a reading list for a query.
- *server*:
  Given a concept graph, run a simple Flask web service to return reading
  lists for queries.

Libraries:
- *techknacq*:
  Core project functionality.
- *mallet*:
  A wrapper for the Mallet Java package, which is used for LDA topic modeling.
- *websearch*:
  Interface for searching the Web with Google or Bing.


## Installation & Requirements

The TechKnAcq toolkit runs in Python 3 on Linux and OS X. We recommend running
it in a Docker virtual machine, as described below. If you want to run it
natively, you will need to install dependencies, including the following.

### Python Dependencies

Install pip3 (Debian/Ubuntu: python3-pip). Use it to install the
required Python packages:

    pip3 install beautifulsoup4 nltk noaho wikipedia gensim
                 networkx pyenchant ftfy flask flask-cors

Patch pyenchant:
  https://github.com/rfk/pyenchant/issues/45

### Precompiled third-party tools

- pdftotext (Ubuntu: poppler-utils)
- MALLET 2.0.7 or 2.0.8. Download and rename the directory to ext/mallet
  (or change the path in 'concept-graph' script).

### Infomap

Download to ext/infomap and compile.

### TechKnAcq Core

    cd ext
    git clone git@github.com:ISI-TechknAcq/techknacq-core.git
    cd techknacq-core
    mvn package
    cd target
    ln -s *jar techknacq-core.jar


## Configuration

Running corpus expansion requires several API keys:
- Put your ScienceDirect API key in ~/.t/sd.txt.
- Put your Bing API key in ~/.t/bing.txt.

Change the file permissions to keep these keys private. They will be mapped
into the Docker virtual machine at runtime, so ~/.t must exist on the machine
you run on.


## Run

### Docker

TechKnAcq is meant to be run from a Docker virtual machine, running on Linux
or macOS. First build it:

    ./build

Then run it:

    ./run

If you have a local directory (e.g., a corpus) that needs to be available in
Docker, map it:

    ./run -v ~/working/corpus:/tmp/corpus

You are now operating in the Docker virtual machine as root.


### Build Corpus

To expand a corpus in ~/corpus, saving the result in ~/expanded, run:

    ./run -v ~/corpus:/tmp/corpus -v ~/expanded:/tmp/expanded
    ./build-corpus --wiki /tmp/corpus /tmp/expanded

The files in the input directory can be in various formats -- ScienceDirect
XML files, BioC XML files, plain text, or the JSON corpus format used for this
project. The output directory will be populated with JSON files that can be
used for generating a concept graph.

The various corpus expansion methods (e.g., '--wiki' above) can be
specified on the command line. Run `./build-corpus --help` to see a full
list.


### Concept Graph

To generate a concept graph from a (possibly expanded) corpus in ~/corpus,
run:

    ./run -v ~/corpus:/tmp/corpus
    ./concept-graph /tmp/corpus

The corpus directory specified should contain JSON files like those produced
by build-corpus. If you have an existing Mallet LDA topic model you'd like
to reuse, specify it second:

   ./concept-graph [corpus dir] [topic model path+prefix]

Since multiple topic models might be in the same directory, the unique prefix
is specified in addition to the path, e.g.,

   ./concept-graph ~/shared/techknacq/Corpora/NLP/current/json/ \
                   ~/scratch/M1/mallet-26205-

With the topic model you can include a topic score file and a topic name file.
These are not generated by default since the topic scoring requires model
generation that is not currently included and the topic naming is currently
manual. These files should have the same prefix as the topic model but end in
'scores.txt' and 'names.csv'. The scores file is one score (float) per line
corresponding to the topics. The names.csv file has lines of format
Topicnum,Name.

The computation of pedagogical roles for documents is not part of this
pipeline, but if a file of these annotations exists with the name
'pedagogical-roles.txt', it will be read by `Corpus.read_roles()` and marked
in the concept graph.


### Reading List

    ./reading-list [concept graph] [query terms]

The concept graph should be a JSON file produced by the concept-graph script
above.


### Server

The techknacq-tk server is a backend that can be used with the web application
in the techknacq-server repository. It is run as:

    ./server [concept graph] ([port])


## Acknowledgments

This research is based upon work supported in part by the Office of the
Director of National Intelligence (ODNI), Intelligence Advanced Research
Projects Activity (IARPA), via Air Force Research Laboratory (AFRL). The views
and conclusions contained herein are those of the authors and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of ODNI, IARPA, AFRL, or the U.S. Government. The
U.S. Government is authorized to reproduce and distribute reprints for
Governmental purposes notwithstanding any copyright annotation thereon.
