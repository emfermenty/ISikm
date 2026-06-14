import os
import sys

# Добавляем корень проекта в sys.path и переходим в него,
# чтобы app.py мог найти model.pkl и model_meta.json по относительным путям.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)
