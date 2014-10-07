DOCS = cosmic_cal.pdf

FIGS = 

BIBLIO = masterbib.bib

BYPROD = *.aux *.log *.out *.dvi *.bbl *.blg __tmp

default: git-tag $(DOCS)

.PHONY : git-tag

git-tag:
	./git-tag.sh

cosmic_cal.pdf: cosmic_cal.tex $(FIGS) $(BIBLIO)
	pdflatex $(@:.pdf=) && bibtex $(@:.pdf=) && pdflatex $(@:.pdf=) && pdflatex $(@:.pdf=) && pdflatex $(@:.pdf=)

clean:
	rm -f $(DOCS) $(BYPROD)
