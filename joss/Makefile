PAPER = paper

.PHONY: all clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	lualatex $(PAPER)
	lualatex $(PAPER)

clean:
	rm -f *.aux *.log *.out *.toc *.pdf
