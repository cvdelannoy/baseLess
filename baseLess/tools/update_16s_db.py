import os
from pathlib import Path
from git import Repo
from baseLess.low_requirement_helper_functions import parse_output_path
__baseLess__ = str(Path(__file__).resolve().parents[1]) + '/'
db_git_url = 'https://github.com/cvdelannoy/baseLess_16s_db.git'

def main(_):
    db_location = __baseLess__ + 'data/16s_nns/'
    if Path(db_location + '.git/').exists():
        repo = Repo(db_location)
    else:
        parse_output_path(db_location, clean=True)
        repo = Repo.clone_from(db_git_url, db_location)
    print(f'16S db successfully pulled, updated to {str(repo.branches[0].commit)}')