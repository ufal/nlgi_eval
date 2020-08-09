#!/usr/bin/env python3

import json
import torch

from argparse import ArgumentParser
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tgen.data import DA

from external.e2e_metrics_tsv import read_tsv
from external.webnlg_entry import Entry, Triple


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


    def check_inst(self, instance):
        # XXX

        inputs = self.tokenizer("Giraffe offers French food. Giraffe is not family-friendly. </s></s> Giraffe serves French food in the riverside and is not family-friendly. ", return_tensors="pt")
        outputs = self.model(**inputs, labels=labels)
        torch.nn.Softmax(dim=1)(outputs[1])

    def parse_e2e(self, fname):
        mrs, sents = read_tsv(fname)
        mrs = [self.da_to_entry(idx, DA.parse_diligent_da(mr)) for idx, mr in enumerate(mrs)]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]

    def da_to_entry(self, eid, da):
        name = [dai for dai in da if dai.slot == 'name'][0].value
        triples = [Triple(name, dai.slot, dai.value) for dai in da if dai.slot != 'name']
        return Entry(eid=eid, size=len(da) - 1, category='E2E', originaltripleset=triples,
                     modifiedtripleset=triples, entitymap=None, lexEntries=None)


    def parse_webnlg(self, fnames):
        # XXX
        # this will be two files -- input & output
        # need to split them
        pass


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
