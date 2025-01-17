./clean.sh
pdflatex Main_Article.tex
bibtex Main_Article
pdflatex Main_Article.tex
pdflatex Main_Article.tex

rm *.out *.aux *.blg *.log
