#!/bin/bash
git clone https://github.com/AP-2020-1S/covid-19-analiticap analiticap
ls -a
cd \analiticap
python3 ejecucionpy.py
git status
git config user.email "you@example.com"
git config user.name "Your Name"
git add .
git commit 
git push origin
