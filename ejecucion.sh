#!/bin/bash
git clone https://github.com/AP-2020-1S/covid-19-analiticap analiticap
ls -a
cd \analiticap
python3 ejecucionpyMedellin.py
python3 ejecucionpyBogota.py
python3 ejecucionpyBarranquilla.py
python3 ejecucionpyCartagena.py
python3 ejecucionpyCali.py
git status
git config user.email "you@example.com"
git config user.name "Your Name"
git add .
git commit -m "Final"
git push origin
