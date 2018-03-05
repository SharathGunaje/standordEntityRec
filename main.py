from __future__ import print_function

import os
import pickle
from argparse import ArgumentParser
from platform import system
from subprocess import Popen
from sys import argv
from sys import stderr

IS_WINDOWS = True if system() == 'Windows' else False
JAVA_BIN_PATH = 'java.exe' if IS_WINDOWS else 'java'
STANFORD_NER_FOLDER = 'stanford-ner'


def arg_parse():
    arg_p = ArgumentParser('Stanford NER Python Wrapper')
    arg_p.add_argument('-f', '--filename', type=str, default=None)
    arg_p.add_argument('-v', '--verbose', action='store_true')
    return arg_p


def debug_print(log, verbose):
    if verbose:
        print(log)


def process_entity_relations(entity_relations_str, verbose=True):
    # format is ollie.
    entity_relations = list()
    for s in entity_relations_str:
        entity_relations.append(s[s.find("(") + 1:s.find(")")].split(';'))
    return entity_relations


def stanford_ner(filename, verbose=True, absolute_path=None):
    out = 'out.txt'

    command = ''
    if absolute_path is not None:
        command = 'cd {};'.format(absolute_path)
    else:
        filename = '../{}'.format(filename)

    command += 'cd {}; {} -mx1g -cp "*:lib/*" edu.stanford.nlp.ie.NERClassifierCombiner ' \
               '-ner.model classifiers/english.all.3class.distsim.crf.ser.gz ' \
               '-outputFormat tabbedEntities -textFile {} > ../{}' \
        .format(STANFORD_NER_FOLDER, JAVA_BIN_PATH, filename, out)

    if verbose:
        debug_print('Executing command = {}'.format(command), verbose)
        java_process = Popen(command, stdout=stderr, shell=True)
    else:
        java_process = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ner exited with a non-zero code status.'

    if absolute_path is not None:
        out = absolute_path + out

    with open(out, 'r') as output_file:
        results_str = output_file.readlines()
    os.remove(out)

    results = []
    for res in results_str:
        if len(res.strip()) > 0:
            split_res = res.split('\t')
            entity_name = split_res[0]
            entity_type = split_res[1]

            if len(entity_name) > 0 and len(entity_type) > 0:
                results.append([entity_name.strip(), entity_type.strip()])

    if verbose:
        pickle.dump(results_str, open('out.pkl', 'wb'))
        debug_print('wrote to out.pkl', verbose)
    return results


def main(args):
    arg_p = arg_parse().parse_args(args[1:])
    filename = arg_p.filename
    verbose = arg_p.verbose
    debug_print(arg_p, verbose)
    if filename is None:
        print('please provide a text file containing your input. Program will exit.')
        exit(1)
    if verbose:
        debug_print('filename = {}'.format(filename), verbose)
    entities = stanford_ner(filename, verbose)
    print('\n'.join([entity[0].ljust(20) + '\t' + entity[1] for entity in entities]))

if __name__ == '__main__':
    exit(main(argv))
