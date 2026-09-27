when Opening a New Terminal:
source venv/bin/activate

Every time you run pip install , Python installs that library inside your venv/ folder.
Running pip freeze > requirements.txt takes a snapshot of all currently installed libraries and overwrites requirements.txt with the updated list:
pip freeze > requirements.txt


GITHUB commit progress:
   git add .
   git commit -m "NOTESSS"
   git push