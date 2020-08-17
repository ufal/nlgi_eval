#!/usr/bin/env python3

import json
import torch
import re
import numpy as np
import pandas as pd
import sklearn

from argparse import ArgumentParser
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tgen.data import DA
from logzero import logger

from external.e2e_metrics_tsv import read_tsv
from external.webnlg_entry import Triple
import external.webnlg_parser as webnlg_parser


TEMPLATE_PATHS = {
    'e2e': 'fusenlg/data/e2e/templates_basic.json',
    'webnlg': 'fusenlg/data/webnlg/templates.json',
}

# the templates are slightly different from what E2E evaluation produced => remap
TEMPLATE_REMAP = {
    'eat_type': 'eatType',
    'rating': 'customer rating',
    'family_friendly': 'familyFriendly',
    'price_range': 'priceRange',
}


class TripleJSONEnc(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Triple):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


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
            self.check_with_gold = self.check_with_gold_webnlg
        elif file_format == 'e2e':
            self.parse_data = self.parse_e2e
            self.check_with_gold = self.check_with_gold_e2e

    def triples_to_templates(self, tripleset):
        output = []
        for triple in tripleset:
            if triple.predicate not in self.templates:
                # if template isn't found, check remapping first
                if triple.predicate in TEMPLATE_REMAP and TEMPLATE_REMAP[triple.predicate] in self.templates:
                    template = self.templates[TEMPLATE_REMAP[triple.predicate]]
                else:  # if remapping doesn't work either, create a backoff template
                    template = 'The %s of <subject> is <object>.' % triple.predicate
                    template = re.sub('([a-z])([A-Z])', r"\1 \2", template)  # get rid of camel case
                    logger.warn('Created backoff template for %s' % (triple.predicate))
            else:  # take the template
                template = self.templates[triple.predicate]
            if isinstance(template, dict):
                template = template[triple.object]
            if isinstance(template, list):
                template = template[0]  # XXX don't take the first, but the best template
            template = template.replace('<subject>', triple.subject)
            obj_str = re.sub('^["\'](.*)["\']$', r'\1', triple.object)  # remove quotes around values
            template = template.replace('<object>', obj_str)
            template = template.replace('_', ' ')
            output.append(template)
        return output

    def roberta_classify(self, a, b):
        inputs = self.tokenizer("%s </s></s> %s" % (a, b), return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # batch size = 1
        outputs = self.model(**inputs, labels=labels)
        outputs = torch.nn.Softmax(dim=1)(outputs[1]).detach().numpy()[0]
        return outputs

    def check_inst(self, mr, sent):
        templs = self.triples_to_templates(mr)

        logger.debug("%s\nTEMP: %s\nSENT: %s" % ("  ++  ".join([str(t) for t in mr]), ' '.join(templs), sent))
        # mr -> sent
        mr2sent = self.roberta_classify(' '.join(templs), sent)
        raw_results = {'mr2sent': "C: %.4f N: %.4f E: %.4f" % tuple(mr2sent)}
        logger.debug("--> " + raw_results['mr2sent'])
        mr2sent = ['C', 'N', 'E'][np.argmax(mr2sent)]
        # sent -> mr
        sent2mr = self.roberta_classify(sent, ' '.join(templs))
        raw_results['sent2mr'] = "C: %.4f N: %.4f E: %.4f" % tuple(sent2mr)
        logger.debug("<-- " + raw_results['sent2mr'])
        sent2mr = ['C', 'N', 'E'][np.argmax(sent2mr)]

        if sent2mr == 'E' and mr2sent == 'E':
            output = 'OK'
        elif sent2mr == 'E':
            output = 'hallucination'
        elif mr2sent == 'E':
            output = 'omission'
        else:
            output = 'hallucination+omission'
        logger.debug(output)

        # sent -> mr for individual slots
        omitted = []
        for triple, templ in zip(mr, templs):
            sent2triple = self.roberta_classify(sent, templ)
            sent2triple = ['C', 'N', 'E'][np.argmax(sent2triple)]
            if sent2triple != 'E':
                omitted.append(triple)
        if omitted:
            logger.debug('Omitted: %s' % '  ++  '.join([str(t) for t in omitted]))
            # override the global decision -- this is more fine-grained
            output = 'hallucination+omission' if 'hallucination' in output else 'omission'

        return output, omitted, raw_results

    def parse_e2e(self, fname):
        mrs, sents = read_tsv(fname)
        mrs = [self.da_to_triples(DA.parse_diligent_da(mr)) for mr in mrs]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]

    def da_to_triples(self, da):
        # convert a TGen DA into WebNLG style triples
        name = [dai for dai in da if dai.slot == 'name'][0].value
        return [Triple(name, dai.slot, dai.value) for dai in da if dai.slot != 'name']

    def parse_webnlg(self, fnames):
        if ',' in fnames:
            # split input & output filenames
            input_fname, output_fname = fnames.split(',')
            # parse input XML triples
            mrs = list(webnlg_parser.parse(input_fname))
            mrs = [mr.modifiedtripleset for mr in mrs]
            # read output sentences
            with open(output_fname, 'r', encoding='UTF-8') as fh:
                sents = [line.strip() for line in fh.readlines()]
        else:
            data = pd.read_csv(fnames, sep=',', encoding='UTF-8').to_dict('records')
            mrs = [[Triple.parse(t) for t in inst['mr'].split('<br>')] for inst in data]
            sents = [inst['text'] for inst in data]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]

    def eval_file(self, in_fname):
        """Main method. Evaluate a given file."""
        data = self.parse_data(in_fname)
        outputs = []
        for mr, sent in data:
            result, omitted, raw_results = self.check_inst(mr, sent)
            outputs.append({'mr': mr,
                            'sent': sent,
                            'result': result,
                            'omitted': omitted,
                            'raw_results': raw_results})

        return outputs

    def check_with_gold_webnlg(self, preds, gold_fname):
        """Evaluation for WebNLG (against the "semantics" column in human eval)."""
        golds = pd.read_csv(gold_fname, sep=',', encoding='UTF-8').to_dict('records')
        for gold in golds:
            gold['mr'] = [Triple.parse(t) for t in gold['mr'].split('<br>')]
            gold['sent'] = gold['text']
            gold['result'] = 'OK' if gold['semantics'] >= 2.5 else 'not OK'

        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            # adding debug info
            pred['gold_result'] = gold['result']
            if pred['result'] != gold['result']:
                pred['error'] = True
            pred['gold_human_rating'] = gold['semantics']
            pred['detailed_result'] = pred['result']
            pred['result'] = 'OK' if pred['result'] == 'OK' else 'not OK'

        # metrics
        y_pred = [inst['result'] for inst in preds]
        y_gold = [inst['result'] for inst in golds]
        acc = sklearn.metrics.accuracy_score(y_gold, y_pred)
        conf_matrix = sklearn.metrics.confusion_matrix(y_gold, y_pred, labels=['OK', 'not OK'])
        conf_matrix = pd.DataFrame(conf_matrix, index=['g_OK', 'g_not'], columns=['p_OK', 'p_not'])

        logger.info('Accuracy: %.4f' % acc)
        logger.info('Confusion matrix:\n%s' % str(conf_matrix))

    def check_with_gold_e2e(self, preds, gold_fname):
        """Evaluation for E2E (against the slot error automatic script predictions)."""
        golds = pd.read_csv(gold_fname, sep='\t', encoding='UTF-8').to_dict('records')
        # adapt format to our output
        for gold in golds:
            gold['mr'] = self.da_to_triples(DA.parse_diligent_da(gold['MR']))
            gold['sent'] = gold['output']
            result = 'OK'
            if gold['added']:
                result = 'hallucination'
            if gold['missing']:
                result = 'hallucination+omission' if gold['added'] else 'omission'
            gold['result'] = result

        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            # adding debug info
            pred['gold_result'] = gold['result']
            if pred['result'] != gold['result']:
                pred['error'] = True
            pred['gold_diff'] = json.loads(gold['diff'])

        # metrics
        y_pred = [inst['result'] for inst in preds]
        y_gold = [inst['result'] for inst in golds]
        acc = sklearn.metrics.accuracy_score(y_gold, y_pred)
        conf_matrix = sklearn.metrics.confusion_matrix(y_gold, y_pred, labels=['OK', 'hallucination', 'omission', 'hallucination+omission'])
        conf_matrix = pd.DataFrame(conf_matrix, index=['g_OK', 'g_hal', 'g_om', 'g_h+o'], columns=['p_OK', 'p_hal', 'p_om', 'p_h+o'])

        logger.info('Accuracy: %.4f' % acc)
        logger.info('Confusion matrix:\n%s' % str(conf_matrix))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--type', '-t', choices=['webnlg', 'e2e'], help='File format/domain templates setting', required=True)
    ap.add_argument('--eval', '-e', action='store_true', help='Input file has gold-standard predictions for evaluation')
    ap.add_argument('input_file', type=str, help='Input file(s) to check')
    ap.add_argument('output_file', type=str, help='Output file')

    args = ap.parse_args()
    evaluator = Evaluator(args.type)
    predictions = evaluator.eval_file(args.input_file)

    if args.eval:
        evaluator.check_with_gold(predictions, args.input_file)

    with open(args.output_file, 'w', encoding='UTF-8') as fh:
        json.dump(predictions, fh, ensure_ascii=False, indent=4, cls=TripleJSONEnc)
