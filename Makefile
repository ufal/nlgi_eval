SHELL=/bin/bash

MEM = 12g
CPUS = 4
GPUMEM = 8g
GPUS = 1

# XXX Replace this with your batch queue engine system command !
QP = --engine=sge --queue=gpu*.q
ifdef LOCAL
  QP = --engine=console
endif
QSUBMIT = qsubmit --logdir '$(TRY_DIR)' --mem $(MEM) --cores $(CPUS) --gpus $(GPUS) --gpu-mem $(GPUMEM) --jobname N.$(TRY_NUM).$(RUN_NAME) $(QP) $(QHOLD) --
# XXX

ACTIVATE = export PYTHONPATH=""; source ./nli_venv/bin/activate  # XXX source your activate script instead

NLI_EVAL = ./nli_eval.py
GREP_SCORES = ./util/grep_scores.py $$file/{all.tsv.json,webnlg_averaged.csv.json}
NOWRAP = ./util/nowrap.pl
SCOREFILE = SCORE

# Runs directories
RUNS_DIR  := runs# main directory for experiment outputs
TRY_NUM   := $(shell perl -e '$$m=0; for(<$(RUNS_DIR)/*>){/\/(\d+)_/ and $$1 > $$m and $$m=$$1;} printf "%03d", $$m+1;')# experiment number
RUN_NAME  := experiment# default name, to be overridden by targets
DATE      := $(shell date +%Y-%m-%d_%H-%M-%S)
TRY_DIR    = $(RUNS_DIR)/$(TRY_NUM)_$(DATE)_$(RUN_NAME)# experiment main output directory (disregarding random seeds and CV)


ifdef D # Shortcut D -> DESC
  DESC := $(D)
endif
NUM_EXP = 20  # number of experiments to list in `make desc'
ifdef N # Shortcut N -> NUM_EXP
  NUM_EXP := $(N)
endif
ifdef R # Shortcut R -> REFRESH
  REFRESH := $(R)
endif


help:
	@echo 'XXX TODO Read the makefile first.'


# List all experiments, find scores in logs (of various versions)
desc:
	@COLS=`tput cols`; \
	ls -d $(RUNS_DIR)/* | sort | tail -n $(NUM_EXP) | while read file; do \
		if [[ -f $$file/$(SCOREFILE) && -z "$(REFRESH)" ]]; then \
			cat $$file/$(SCOREFILE) | $(NOWRAP) $$COLS; \
			continue ; \
		fi ; \
		echo -ne $$file | sed 's/runs\///;s/_/  /;s/_/ /;s/_.*/: /;s/  20/  /;s/-[0-9][0-9]  /  /;' > $$file/$(SCOREFILE);  \
		$(GREP_SCORES) >> $$file/$(SCOREFILE) ; \
		cat $$file/ABOUT | tr  '\n' ' ' >> $$file/$(SCOREFILE); \
		echo >> $$file/$(SCOREFILE); \
		cat $$file/$(SCOREFILE) | $(NOWRAP) $$COLS; \
	done

desc_wrap:
	@ls -d $(RUNS_DIR)/* | sort | tail -n $(NUM_EXP) | while read file; do \
		cat $$file/$(SCOREFILE) ; \
	done

printgit:
	@git status
	@echo -e "\n*** *** ***\n"
	@git log --pretty=format:"%h - %an, %ar : %s" -1
	@echo -e "\n*** *** ***\n"
	@git diff

printvars:
	$(foreach V, $(sort $(.VARIABLES)), $(if $(filter-out environment% default automatic, $(origin $V)), $(info $V=$($V) ($(value $V)))))

prepare_dir:
	# create the experiment directory, print description and git version
	echo "PREPARE DIR" $(TRY_DIR) $(CV_NUM) $(SEED)
	mkdir -p $(TRY_DIR)
	echo "$(DESC)" > $(TRY_DIR)/ABOUT
	make printvars > $(TRY_DIR)/VARS
	make printgit > $(TRY_DIR)/GIT_VERSION


run: prepare_dir
run:
	$(QSUBMIT) '$(ACTIVATE); \
	for file in data-e2e/*; do \
	    echo ">>>>" $$file "<<<<"; \
	    base=`basename $$file`; \
	    $(NLI_EVAL) -e -t e2e $(PARAM) $$file $(TRY_DIR)/$$base.json; \
	done; \
	for file in data-webnlg/*; do \
	    echo ">>>>" $$file "<<<<"; \
	    $(NLI_EVAL) -e -t webnlg $(PARAM) $$file $(TRY_DIR)/$$base.json; \
	done' 2>&1 | tee $(TRY_DIR)/$(RUN_NAME).log
