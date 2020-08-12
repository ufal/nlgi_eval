#!/usr/bin/env python3

import json
import torch

from argparse import ArgumentParser
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tgen.data import DA
from logzero import logger

from external.e2e_metrics_tsv import read_tsv
from external.webnlg_entry import Entry, Triple
import external.webnlg_parser as webnlg_parser


TEMPLATE_PATHS = {
    'e2e': 'fusenlg/data/e2e/templates_basic.json',
    'webnlg': 'fusenlg/data/webnlg/templates.json',
}


class Evaluator:

    def __init__(self, file_format):

        # load roberta
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

        # load templates
        with open(TEMPLATE_PATHS[file_format], 'r', encoding='UTF-8') as fh:
            self.templates = json.load(fh)
        # set parse method
        if file_format == 'webnlg':
            self.parse_data = self.parse_webnlg
        elif file_format == 'e2e':
            self.parse_data = self.parse_e2e

    def triples_to_templates(self, tripleset):
        output = ''
        for triple in tripleset:
            template = self.templates[triple.predicate]
            if isinstance(template, dict):
                template = template[triple.object]
            if isinstance(template, list):
                template = template[0]  # XXX don't take the first, but the best template
            template = template.replace('<subject>', triple.subject)
            template = template.replace('<object>', triple.object)
            template = template.replace('_', ' ')
            output += (' ' if output else '') + template
        return output

    def roberta_classify(self, a, b):
        inputs = self.tokenizer("%s </s></s> %s" % (a, b), return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # batch size = 1
        outputs = self.model(**inputs, labels=labels)
        outputs = torch.nn.Softmax(dim=1)(outputs[1]).detach().numpy()[0]
        return outputs

    def check_inst(self, instance):
        mr, sent = instance
        templ = self.triples_to_templates(mr.modifiedtripleset)

        logger.debug("%s\nTEMP: %s\nSENT: %s" % (" + ".join([repr(t) for t in mr.modifiedtripleset]), templ, sent))
        # mr -> sent
        outputs = self.roberta_classify(templ, sent)
        logger.debug("--> C: %.4f N: %.4f E: %.4f" % tuple(outputs))
        # sent -> mr
        outputs = self.roberta_classify(sent, templ)
        logger.debug("<-- C: %.4f N: %.4f E: %.4f" % tuple(outputs))

    def parse_e2e(self, fname):
        mrs, sents = read_tsv(fname)
        mrs = [self.da_to_entry(idx, DA.parse_diligent_da(mr)) for idx, mr in enumerate(mrs)]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]

    def da_to_entry(self, eid, da):
        # convert a TGen DA into WebNLG style triples
        name = [dai for dai in da if dai.slot == 'name'][0].value
        triples = [Triple(name, dai.slot, dai.value) for dai in da if dai.slot != 'name']
        return Entry(eid=eid, size=len(da) - 1, category='E2E', originaltripleset=triples,
                     modifiedtripleset=triples, entitymap=None, lexEntries=None)


    def parse_webnlg(self, fnames):
        # split input & output filenames
        input_fname, output_fname = fnames.split(',')
        # parse input XML triples
        mrs = list(webnlg_parser.parse(input_fname))
        # read output sentences
        with open(output_fname, 'r', encoding='UTF-8') as fh:
            sents = [line.strip() for line in fh.readlines()]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]


    def eval_file(self, fname):
        # XXX main method
        data = self.parse_data(fname)

        for instance in data:
            result = self.check_inst(instance)



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--type', '-t', choices=['webnlg', 'e2e'], help='File format/domain templates setting', required=True)
    ap.add_argument('input_file', type=str, help='Input file to check')

    args = ap.parse_args()
    evaluator = Evaluator(args.type)
    evaluator.eval_file(args.input_file)

    # XXX some way of checking the correlations
