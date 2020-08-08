#!/usr/bin/env python3

from argparse import ArgumentParser
from transformers import import RobertaTokenizer, RobertaForSequenceClassification


class Evaluator:

    def __init__(self, file_format):

        # load roberta
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

        # load templates
        # set parse method
        if file_format == 'webnlg':
            # XXX
        elif file_format == 'e2e':
            # XXX




    def check_inst(instance):
        # XXX
        inputs = tokenizer("Giraffe offers French food. Giraffe is not family-friendly. </s></s> Giraffe serves French food in the riverside and is not family-friendly. ", return_tensors="pt")
        outputs = model(**inputs, labels=labels)
        torch.nn.Softmax(dim=1)(outputs[1])

    def parse_e2e(fname):
        # XXX

    def parse_webnlg(fnames):
        # XXX
        # this will be two files -- input & output
        # need to split them


    def eval_file(fname):
        # XXX main method
        data = self.parse_data(fname)


        for instance in data:
            result = check_inst(instance)



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--type', '-t', choices=['webnlg', 'e2e'], help='File format/domain templates setting', required=True)
    ap.add_argument('input_file', type=str, help='Input file to check')

    args = ap.parse_args()
    evaluator = Evaluator(args.type)
    evaluator.eval_file(args.input_file)

    # XXX some way of checking the correlations
