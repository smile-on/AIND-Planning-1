"""
This is a helper to run the searches in batch and automatically collect results in csv file.
I chose to run 4 search methods 6 times for each of 3 cargo problems.

Note, run_search was written by Udacity without thinking of capturing output, mixing run objects with print() in stdout
thus some mess is here as well.
"""

from pathlib import Path
import re
from contextlib import redirect_stdout
from multiprocessing import Pool

from run_search import main as run_search_main

runs = 6 # better be less than num CPUS
problems = ['1', '2', '3' ] # cargo problems
#searches = ['1', '3', '5', '7'] # uninformed search methods: bfs dfs uniform_cost_search greedy_best_first_graph_search h_1 
searches = ['9', '10'] # heuristic search using A* with the "ignore preconditions" and "level-sum" heuristics
log_dir = Path("./log")
results_csv = log_dir / "search_result.csv"
batches = [{"id":n, "log_file": log_dir/ f"search-{n}.log"} for n in range(runs)]

def run_batch(log_file: Path):
    """ executes search metods that dumps output in text file
    """
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            print(f"run_search_main({problems}, {searches})")
            run_search_main(problems, searches)

def log_header():
    with open(results_csv, "w+") as result_f:
        result_f.write("run, problem, method, Expansions, GoalTests, NewNodes, plan_size, time_elapsed,\n")

def log_results(id: int, output: Path):
    """ parse output of an run with few search methods
    log results in format:
    run, problem, method, Expansions, GoalTests, NewNodes, plan_size, time_elapsed,
    """
    problem_re = re.compile("^Solving Air Cargo Problem (\d+) using (\w+) with (\w+)*.")
    search_re = re.compile("^Plan length: (\d+)  Time elapsed in seconds: (\d+\.?\d*)*.")
    with open(results_csv, "a") as result_f:
        with open(output) as output_f:
            # parsing output
            for line in output_f:
                m = problem_re.match(line)
                if m: # "Solving Air Cargo Problem 1 using depth_first_graph_search..."
                    problem = m.group(1)
                    method = m.group(2)+'_'+m.group(3)
                    result_f.write(f"{id},{problem},{method},")
                if line.startswith("Expansions   Goal Tests   New Nodes"):
                    # parse next line  " 43 56 180 " 
                    vals = output_f.readline().split()
                    result_f.write(f"{vals[0]},{vals[1]},{vals[2]},")
                m = search_re.match(line)
                if m: # "Plan length: 12  Time elapsed in seconds: 0.024"
                    plan_size = m.group(1)
                    time_elapsed = m.group(2)
                    result_f.write(f"{plan_size},{time_elapsed}")
                    result_f.write("\n")

                    
if __name__=="__main__":
    if not log_dir.exists():
        print(f'ERR: log dir does not exist {log_dir}') 
    else:
        # batches are exec in parallel
        with Pool(runs) as p:
            p.map( run_batch, [b["log_file"] for b in batches] )
        # collect results
        log_header()
        for b in batches:
            log_file = b["log_file"]
            id = b["id"]
            log_results(id, log_file)
        print('OK: all batches done, normal exit.')

