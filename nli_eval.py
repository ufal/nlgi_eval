#!/usr/bin/env python3

import json
import torch
import re
import numpy as np
import pandas as pd
import sklearn
import scipy

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

    def __init__(self, file_format, use_templates=True):

        self.use_gpu = torch.cuda.is_available()
        logger.debug('Use GPU: %r' % self.use_gpu)

        # load roberta
        logger.debug('Loading models...')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
        if self.use_gpu:
            self.model.to('cuda')

        # load templates
        if use_templates:
            logger.debug('Loading templates...')
            with open(TEMPLATE_PATHS[file_format], 'r', encoding='UTF-8') as fh:
                self.templates = json.load(fh)
        else:
            self.templates = {}
        # set parse method
        if file_format == 'webnlg':
            self.parse_data = self.parse_webnlg
            self.check_with_gold = self.check_with_gold_webnlg
        elif file_format == 'e2e':
            self.parse_data = self.parse_e2e
            self.check_with_gold = self.check_with_gold_e2e
        logger.debug('Ready.')

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
                    self.templates[triple.predicate] = template
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
        if self.use_gpu:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        _, output = self.model(**inputs, labels=labels)  # ignoring loss
        if self.use_gpu:
            output = output.cpu()
        output = torch.nn.Softmax(dim=1)(output).detach().numpy()[0]
        return output

    def check_inst(self, mr, sent):
        templs = self.triples_to_templates(mr)

        logger.debug("%s\nTEMP: %s\nSENT: %s" % ("  ++  ".join([str(t) for t in mr]), ' '.join(templs), sent))
        ent_confs = []
        # mr -> sent
        mr2sent = self.roberta_classify(' '.join(templs), sent)
        raw_results = {'mr2sent': "C: %.4f N: %.4f E: %.4f" % tuple(mr2sent)}
        logger.debug("--> " + raw_results['mr2sent'])
        ent_confs.append(float(mr2sent[2]))
        mr2sent = ['C', 'N', 'E'][np.argmax(mr2sent)]
        # sent -> mr
        sent2mr = self.roberta_classify(sent, ' '.join(templs))
        raw_results['sent2mr'] = "C: %.4f N: %.4f E: %.4f" % tuple(sent2mr)
        logger.debug("<-- " + raw_results['sent2mr'])
        ent_confs.append(float(sent2mr[2]))
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
            ent_confs.append(float(sent2triple[2]))
            sent2triple = ['C', 'N', 'E'][np.argmax(sent2triple)]
            if sent2triple != 'E':
                omitted.append(triple)

        if omitted:
            logger.debug('Omitted: %s' % '  ++  '.join([str(t) for t in omitted]))
            # override the global decision -- this is more fine-grained
            output = 'hallucination+omission' if 'hallucination' in output else 'omission'

        return output, min(ent_confs), omitted, raw_results

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
            result, OK_conf, omitted, raw_results = self.check_inst(mr, sent)
            outputs.append({'mr': mr,
                            'sent': sent,
                            'result': result,
                            'OK_confidence': OK_conf,
                            'omitted': omitted,
                            'raw_results': raw_results})

        return outputs

    def compute_metrics(self, y_pred, y_gold, fine=False):
        """Compute accuracy and confusion matrix."""
        labels_order = ['OK', 'not OK']
        row_labels=['g_OK', 'g_not']
        col_labels=['p_OK', 'p_not']
        if fine:
            labels_order = ['OK', 'hallucination', 'omission', 'hallucination+omission']
            row_labels = ['g_OK', 'g_hal', 'g_om', 'g_h+o']
            col_labels = ['p_OK', 'p_hal', 'p_om', 'p_h+o']

        acc = sklearn.metrics.accuracy_score(y_gold, y_pred)
        conf_matrix = sklearn.metrics.confusion_matrix(y_gold, y_pred, labels=labels_order)
        conf_matrix = pd.DataFrame(conf_matrix, index=row_labels, columns=col_labels)

        return acc, conf_matrix

    def check_with_gold_webnlg(self, preds, gold_fname):
        """Evaluation for WebNLG (against the "semantics" column in human eval)."""
        golds = pd.read_csv(gold_fname, sep=',', encoding='UTF-8').to_dict('records')
        for gold in golds:
            gold['mr'] = [Triple.parse(t) for t in gold['mr'].split('<br>')]
            gold['sent'] = gold['text']

        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            # adding debug info
            pred['gold_human_rating'] = gold['semantics']
            if (pred['result'] == 'OK') != (gold['semantics'] >= 2.5):
                pred['error'] = True

        results = {'predictions': preds}
        # metrics
        for threshold in [2.5, 2.0]:
            y_pred = ['OK' if inst['result'] == 'OK' else 'not OK' for inst in preds]
            y_gold = ['OK' if inst['semantics'] >= 2.5 else 'not OK' for inst in golds]
            acc, conf_matrix = self.compute_metrics(y_pred, y_gold)
            logger.info('Threshold %.1f -- Accuracy: %.4f' % (threshold, acc))
            logger.info('Confusion matrix:\n%s' % str(conf_matrix))
            results['metrics @ %.1f' % threshold] = {'accuracy': acc, 'conf_matrix': conf_matrix.to_dict('index')}

        # correlation with humans
        conf_rho, conf_rho_p = scipy.stats.spearmanr([inst['OK_confidence'] for inst in preds],
                                                     [inst['semantics'] for inst in golds])
        logger.info('Spearman correlation of OK_confidence with human ratings: %.4f (p=%.4f)'
                    % (conf_rho, conf_rho_p))
        results['OK_correlation'] = {'rho': conf_rho, 'p_value': conf_rho_p}
        return results

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

        results = {'predictions': preds}
        # metrics + rough metrics
        y_pred = [inst['result'] for inst in preds]
        y_gold = [inst['result'] for inst in golds]
        acc, conf_matrix = self.compute_metrics(y_pred, y_gold, fine=True)
        logger.info('Accuracy: %.4f' % acc)
        logger.info('Confusion matrix:\n%s' % str(conf_matrix))
        results['metrics_fine'] = {'accuracy': acc, 'conf_matrix': conf_matrix.to_dict('index')}

        y_pred = ['OK' if inst == 'OK' else 'not OK' for inst in y_pred]
        y_gold = ['OK' if inst == 'OK' else 'not OK' for inst in y_gold]
        acc, conf_matrix = self.compute_metrics(y_pred, y_gold, fine=False)
        logger.info('Accuracy: %.4f' % acc)
        logger.info('Confusion matrix:\n%s' % str(conf_matrix))
        results['metrics_rough'] = {'accuracy': acc, 'conf_matrix': conf_matrix.to_dict('index')}
        return results


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--type', '-t', choices=['webnlg', 'e2e'], help='File format/domain templates setting', required=True)
    ap.add_argument('--eval', '-e', action='store_true', help='Input file has gold-standard predictions for evaluation')
    ap.add_argument('--no-templates', dest='templates', action='store_false', help='Do not use any preloaded templates, use backoff only')
    ap.add_argument('input_file', type=str, help='Input file(s) to check')
    ap.add_argument('output_file', type=str, help='Output file')

    args = ap.parse_args()

    logger.debug('Starting...')
    evaluator = Evaluator(args.type, args.templates)
    predictions = evaluator.eval_file(args.input_file)

    logger.debug('Evaluation...')
    if args.eval:
        predictions = evaluator.check_with_gold(predictions, args.input_file)

    logger.debug('Writing output...')
    with open(args.output_file, 'w', encoding='UTF-8') as fh:
        json.dump(predictions, fh, ensure_ascii=False, indent=4, cls=TripleJSONEnc)
    logger.debug('Done.')
