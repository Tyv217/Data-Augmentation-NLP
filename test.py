import subprocess
from datetime import datetime, timedelta

def git_log_summary():
    two_months_ago = datetime.now() - timedelta(days=60)
    git_log = subprocess.check_output(["git", "log", "--numstat", "--pretty=format:'%H %ct'", "--since='2 months ago'"]).decode("utf-8")
    lines = git_log.split('\n')
    commit_summaries = []
    commit = None
    for line in lines:
        if line.startswith("'"):
            commit_hash, commit_time = line[1:-1].split(' ')
            commit_datetime = datetime.fromtimestamp(int(commit_time))
            if commit_datetime > two_months_ago:
                if commit:
                    commit_summaries.append(commit)
                commit = {'commit_hash': commit_hash, 'changes': 0, 'files_changed': 0}
        elif line:
            commit['files_changed'] += 1
            if line.split()[0] != '-':
                commit['changes'] += int(line.split()[0])
            if line.split()[1] != '-':
                commit['changes'] += int(line.split()[1])
    if commit:
        commit_summaries.append(commit)
    commit_summaries.sort(key=lambda c: c['files_changed'], reverse=True)
    return commit_summaries

x = git_log_summary()
for i in x:
    print(i)