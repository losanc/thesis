DOCNAME=thesis

all: report

.PHONY: clean

report:
	pdflatex -halt-on-error  $(DOCNAME).tex 
	bibtex $(DOCNAME).aux
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex

view: report
	open $(DOCNAME).pdf

clean:
	rm *.blg *.bbl *.aux *.log
